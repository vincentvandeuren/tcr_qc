//! Bulk discovery of near-identical junctions within a repertoire.
//!
//! Finds every pair of rows whose sequences are within `max_distance` of each
//! other, restricted to rows agreeing on the `by` columns. The work is done by
//! the `symscan` crate, which implements symmetric deletion lookup: index each
//! string by all of its single-character deletion variants, then any two
//! strings sharing a variant differ by at most one edit. symscan replaces the
//! usual hash map over variants with a sort-merge join, trading an O(log N)
//! factor for cache locality and parallelism, which is the right trade for
//! searching one large collection rather than many small lookups.
//!
//! WHY NOT THE OBVIOUS ALTERNATIVE. The textbook trick for Hamming-1 pairs is
//! to bucket on the even- and odd-indexed characters: two strings one
//! substitution apart must agree on one of the two halves. It is simple and
//! complete, but every bucket still needs an all-pairs comparison, so the
//! comparison count grows quadratically in bucket size — measured at 4x per
//! doubling of input on real CDR3s, against 2x for deletion variants. The
//! crossover is well below cohort scale, so the parity version is not worth
//! carrying.
//!
//! WHY GROUPING IS DONE HERE AND NOT BY THE CALLER. Requiring neighbours to
//! share a V gene discards a lot: on real TRB data 85% of Hamming-1 junction
//! pairs are between rows with *different* `v_call`s, because `CASS…` is
//! produced by dozens of TRBV genes and V identity simply is not determined
//! within the junction. Searching pooled and filtering afterwards would spend
//! most of its time on candidates that are then thrown away, so the partition
//! happens before the search.
//!
//! Distance 0 is reported, not suppressed. When rows are keyed by nucleotide
//! clonotype, two rows sharing a `junction_aa` are independent recombination
//! events that converged on the same amino-acid sequence — real observations
//! rather than duplicates. The caller decides what to do with them; the
//! `distance` column is what makes that decision possible.

use std::sync::atomic::{AtomicU32, Ordering};

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use rayon::prelude::*;
use symscan::{get_hamming_neighbors_within, get_neighbors_within, NeighborPairs};

/// Row indices of one group, paired with the sequences to search.
///
/// Null sequences are dropped rather than passed through: symscan has no
/// notion of a missing string, and a null junction cannot be anyone's
/// neighbour. `rows` keeps the original frame position so local results can be
/// lifted back to global indices.
struct Group<'a> {
    rows: Vec<u32>,
    seqs: Vec<&'a str>,
}

/// Pairs of rows within `max_distance` of each other, sharing every `by` column.
///
/// Returns a frame of `idx_a`, `idx_b` (`u32` row positions into the input,
/// always `idx_a < idx_b`) and `distance`. Each pair appears once.
///
/// `distance` is widened to `u32` from symscan's `u8`: polars is built here
/// with `default-features = false`, so the `u8` dtype is not available, and
/// turning it on for one column is a larger change to the crate than the three
/// bytes per row are worth.
///
/// Row positions rather than identifiers: the caller already holds the frame
/// and can gather any column it likes, which keeps this function from having
/// to care whether the identifier is an integer, a string, or absent.
///
/// `metric` is `"hamming"` (equal lengths only, substitutions) or
/// `"levenshtein"` (also insertions and deletions). Hamming is the right
/// default for junction comparison, where an indel changes the length and is a
/// different event from a substitution.
#[pyfunction]
#[pyo3(signature = (df, sequence="junction_aa", by=vec!["v_call".to_string()], max_distance=1, metric="hamming"))]
pub fn hd1_pairs(
    py: Python,
    df: PyDataFrame,
    sequence: &str,
    by: Vec<String>,
    max_distance: u8,
    metric: &str,
) -> PyResult<PyDataFrame> {
    let err = |e: PolarsError| pyo3::exceptions::PyValueError::new_err(e.to_string());
    let df: DataFrame = df.into();

    let hamming = parse_metric(metric)?;

    // rechunk so the string data is one contiguous buffer: the &str handed to
    // symscan borrow straight into it, so no sequence is ever copied.
    let seq_col = df
        .column(sequence)
        .map_err(err)?
        .str()
        .map_err(err)?
        .rechunk();

    let groups = partition(&df, &seq_col, &by).map_err(err)?;

    // Each group is an independent search. symscan parallelises internally as
    // well; rayon's work stealing means the nesting does not oversubscribe,
    // and it keeps many small groups from running one at a time.
    //
    // `detach` rather than `allow_threads`: same thing, but the latter is
    // deprecated in pyo3 0.27. Either way the GIL is released so other Python
    // threads run while this goes wide.
    let found: Result<Vec<NeighborPairs>, symscan::Error> = py.detach(|| {
        groups
            .par_iter()
            .map(|g| {
                if hamming {
                    get_hamming_neighbors_within(&g.seqs, max_distance)
                } else {
                    get_neighbors_within(&g.seqs, max_distance)
                }
            })
            .collect()
    });
    let found = found.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let total: usize = found.iter().map(|f| f.row.len()).sum();
    let mut idx_a: Vec<u32> = Vec::with_capacity(total);
    let mut idx_b: Vec<u32> = Vec::with_capacity(total);
    let mut dists: Vec<u32> = Vec::with_capacity(total);

    for (g, pairs) in groups.iter().zip(found) {
        for ((a, b), d) in pairs.row.iter().zip(pairs.col.iter()).zip(pairs.dists) {
            // Lift group-local positions back to frame rows. symscan orders
            // each pair ascending in its own indexing, which does not survive
            // the lift, so normalise here.
            let (ga, gb) = (g.rows[*a as usize], g.rows[*b as usize]);
            idx_a.push(ga.min(gb));
            idx_b.push(ga.max(gb));
            dists.push(d as u32);
        }
    }

    let out = df![
        "idx_a" => idx_a,
        "idx_b" => idx_b,
        "distance" => dists,
    ]
    .map_err(err)?;
    Ok(PyDataFrame(out))
}

/// How many neighbours each row has, without ever building the edge list.
///
/// Returns a frame aligned to the input: `n_neighbors` (rows within
/// `max_distance`, excluding the row itself) and `n_exact` (the subset at
/// distance 0).
///
/// WHY THIS EXISTS ALONGSIDE `hd1_pairs`. Counting degrees through the pair
/// list means materialising every edge only to aggregate it straight back down
/// to one number per row, and the edge count grows far faster than the row
/// count -- a repertoire with a dense neighbourhood can have many times more
/// edges than clonotypes. Here each group's pairs are consumed as they are
/// produced and dropped, so peak memory is one group's edges rather than the
/// whole cohort's.
///
/// `n_exact` is kept separate because distance 0 means something different
/// from distance 1: independent rearrangements converging on one amino-acid
/// sequence, rather than a one-substitution relative. Collapsing them loses
/// that, and recovering it afterwards would need the edge list this function
/// exists to avoid.
#[pyfunction]
#[pyo3(signature = (df, sequence="junction_aa", by=vec!["v_call".to_string()], max_distance=1, metric="hamming"))]
pub fn neighbor_counts(
    py: Python,
    df: PyDataFrame,
    sequence: &str,
    by: Vec<String>,
    max_distance: u8,
    metric: &str,
) -> PyResult<PyDataFrame> {
    let err = |e: PolarsError| pyo3::exceptions::PyValueError::new_err(e.to_string());
    let df: DataFrame = df.into();
    let hamming = parse_metric(metric)?;
    let height = df.height();

    let seq_col = df
        .column(sequence)
        .map_err(err)?
        .str()
        .map_err(err)?
        .rechunk();
    let groups = partition(&df, &seq_col, &by).map_err(err)?;

    // Atomics rather than per-thread tallies: groups partition the rows, so two
    // threads never touch the same index and the counters are uncontended. The
    // alternative, merging per-group count vectors, would allocate one vector
    // per group for no benefit.
    let total: Vec<AtomicU32> = (0..height).map(|_| AtomicU32::new(0)).collect();
    let exact: Vec<AtomicU32> = (0..height).map(|_| AtomicU32::new(0)).collect();

    let outcome: Result<(), symscan::Error> = py.detach(|| {
        groups.par_iter().try_for_each(|g| {
            let pairs = if hamming {
                get_hamming_neighbors_within(&g.seqs, max_distance)?
            } else {
                get_neighbors_within(&g.seqs, max_distance)?
            };
            for ((a, b), d) in pairs.row.iter().zip(pairs.col.iter()).zip(pairs.dists) {
                let (ga, gb) = (g.rows[*a as usize] as usize, g.rows[*b as usize] as usize);
                total[ga].fetch_add(1, Ordering::Relaxed);
                total[gb].fetch_add(1, Ordering::Relaxed);
                if d == 0 {
                    exact[ga].fetch_add(1, Ordering::Relaxed);
                    exact[gb].fetch_add(1, Ordering::Relaxed);
                }
            }
            Ok(())
        })
    });
    outcome.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let drain = |v: Vec<AtomicU32>| -> Vec<u32> { v.into_iter().map(|a| a.into_inner()).collect() };
    let out = df![
        "n_neighbors" => drain(total),
        "n_exact" => drain(exact),
    ]
    .map_err(err)?;
    Ok(PyDataFrame(out))
}

/// Shared by both entry points so they cannot disagree about metric names.
fn parse_metric(metric: &str) -> PyResult<bool> {
    match metric {
        "hamming" => Ok(true),
        "levenshtein" => Ok(false),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "metric must be 'hamming' or 'levenshtein', got {other:?}"
        ))),
    }
}

/// Split the frame into groups agreeing on every `by` column.
///
/// An empty `by` is a single group over the whole frame, which is what you
/// want when neighbours are allowed to cross every annotation.
fn partition<'a>(
    df: &DataFrame,
    seq_col: &'a StringChunked,
    by: &[String],
) -> PolarsResult<Vec<Group<'a>>> {
    let take = |rows: Vec<u32>| -> Group<'a> {
        let mut kept = Vec::with_capacity(rows.len());
        let mut seqs = Vec::with_capacity(rows.len());
        for r in rows {
            if let Some(s) = seq_col.get(r as usize) {
                kept.push(r);
                seqs.push(s);
            }
        }
        Group { rows: kept, seqs }
    };

    if by.is_empty() {
        return Ok(vec![take((0..df.height() as u32).collect())]);
    }

    let by_cols: Vec<&str> = by.iter().map(|s| s.as_str()).collect();
    let grouped = df.group_by_stable(by_cols)?;
    Ok(grouped
        .get_groups()
        .iter()
        .map(|g| {
            let rows: Vec<u32> = match g {
                GroupsIndicator::Idx((_, idx)) => idx.to_vec(),
                GroupsIndicator::Slice([first, len]) => (first..first + len).collect(),
            };
            take(rows)
        })
        // A group of one cannot contain a pair; skipping them keeps the
        // parallel map from being swamped by singletons, which dominate.
        .filter(|g| g.seqs.len() > 1)
        .collect())
}
