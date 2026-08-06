mod expressions;
mod d_segment;
mod reference_points;
mod junction_trimmer;
mod genes_imgt;
mod franken;

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PolarsAllocator;
use pyo3_polars::derive::polars_expr;
use std::sync::LazyLock;
use std::fmt::Write;


#[pyfunction]
fn times_two_function(x: isize) -> isize {
    x * 2
}

#[pymodule]
fn _internal(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(times_two_function, m)?)?;
    m.add_function(wrap_pyfunction!(count_matches_function, m)?)?;
    m.add_function(wrap_pyfunction!(franken::franken_candidates, m)?)?;
    Ok(())
}

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();


fn count_matches(s1: &str, s2: &str) -> usize {
    s1.chars()
    .zip(s2.chars())
    .take_while(|(a, b)| a == b)
    .count()
}


#[pyfunction]
fn count_matches_function(s1: &str, s2: &str) -> usize {
    count_matches(s1, s2)
}

use std::collections::{hash_map::Entry, BTreeMap, HashMap};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DGeneSubstring {
    pub full_seq: String,
    pub substring: String,
    pub n_d0: usize,
    pub n_d1: usize,
    pub length: usize,
}

// Internal helper: generate all substrings for one sequence (ASCII assumed).
fn generate_substrings(seq: &str, min_length: usize) -> impl Iterator<Item = DGeneSubstring> + '_ {
    let n = seq.len();
    (min_length..=n).flat_map(move |length| {
        (0..=n - length).map(move |start| {
            let substring = seq[start..start + length].to_string();
            let n_d0 = start;
            let n_d1 = n - (start + length);
            DGeneSubstring {
                full_seq: seq.to_string(),
                substring,
                n_d0,
                n_d1,
                length,
            }
        })
    })
}

/// Group unique substrings by length, keeping the entry with the smallest n_d0 + n_d1.
/// Buckets are sorted so you can short-circuit on the first match per length.
pub fn build_substring_index<S: AsRef<str>>(
    seqs: impl IntoIterator<Item = S>,
    min_length: usize,
) -> BTreeMap<usize, Vec<DGeneSubstring>> {
    // Deduplicate: substring -> best DGeneSubstring by (n_d0 + n_d1)
    let mut best_by_sub: HashMap<String, DGeneSubstring> = HashMap::new();

    for seq_s in seqs {
        let seq = seq_s.as_ref();
        if seq.len() < min_length {
            continue;
        }
        for s in generate_substrings(seq, min_length) {
            let key = s.substring.clone();
            let cand_sum = s.n_d0 + s.n_d1;
            match best_by_sub.entry(key) {
                Entry::Vacant(e) => {
                    e.insert(s);
                }
                Entry::Occupied(mut e) => {
                    let existing_sum = e.get().n_d0 + e.get().n_d1;
                    if cand_sum < existing_sum {
                        e.insert(s);
                    } else if cand_sum == existing_sum {
                        // Optional: deterministic tie-breaks (smallest n_d0, then n_d1, then lexicographic)
                        let ex = e.get();
                        let replace = (s.n_d0, s.n_d1, &s.full_seq, &s.substring)
                            < (ex.n_d0, ex.n_d1, &ex.full_seq, &ex.substring);
                        if replace {
                            e.insert(s);
                        }
                    }
                }
            }
        }
    }

    // Group by length in a BTreeMap so we can iterate lengths in descending order.
    let mut by_len: BTreeMap<usize, Vec<DGeneSubstring>> = BTreeMap::new();
    for v in best_by_sub.into_values() {
        by_len.entry(v.length).or_default().push(v);
    }

    // Sort each bucket by increasing deletions, then by n_d0/n_d1/substring for determinism.
    for (_len, vec) in by_len.iter_mut() {
        vec.sort_by(|a, b| {
            (a.n_d0 + a.n_d1)
                .cmp(&(b.n_d0 + b.n_d1))
                .then_with(|| a.n_d0.cmp(&b.n_d0))
                .then_with(|| a.n_d1.cmp(&b.n_d1))
                .then_with(|| a.substring.cmp(&b.substring))
        });
    }

    by_len
}

/// Find the best D-segment substring:
/// - Longest length that occurs in `non_templated`.
/// - For that length, minimal (n_d0 + n_d1). Because buckets are sorted, we can return on first match.
///
/// Returns a reference into `index`. Use Option<&DGeneSubstring> rather than a sentinel struct.
pub fn get_most_likely_d_segment<'a>(
    non_templated: &str,
    index: &'a BTreeMap<usize, Vec<DGeneSubstring>>,
) -> Option<&'a DGeneSubstring> {
    if index.is_empty() {
        return None;
    }

    let max_len_allowed = non_templated.len();

    // Iterate lengths in descending order, but only up to max_len_allowed.
    // BTreeMap::range(..=x) gives ascending; rev() makes it descending.
    for (_len, bucket) in index.range(..=max_len_allowed).rev() {
        // Bucket is pre-sorted by fewest deletions, so first match is optimal for this length.
        if let Some(found) = bucket.iter().find(|d| non_templated.contains(&d.substring)) {
            return Some(found);
        }
    }

    None
}

static HUMAN_TRBD_SUBSTRINGS: LazyLock<BTreeMap<usize, Vec<DGeneSubstring>>> = LazyLock::new(|| {
    build_substring_index(
        ["gggacagggggc", "gggactagcggggggg", "gggactagcgggaggg"],
        3,
    )
});


fn trim_d_segment_for_human(non_templated: &str, output: &mut String) -> () {
    if let Some(best) = get_most_likely_d_segment(non_templated, &HUMAN_TRBD_SUBSTRINGS) {
        write!(output, "{}", best.substring).unwrap();
    } else {
        output.clear();
    }
}

#[polars_expr(output_type=String)]
fn get_most_likely_d_segment_for_human_polars(inputs: &[Series]) -> PolarsResult<Series> {
    let ca: &StringChunked = inputs[0].str()?;
    let out: StringChunked = ca.apply_into_string_amortized(trim_d_segment_for_human);
    Ok(out.into_series())
}

