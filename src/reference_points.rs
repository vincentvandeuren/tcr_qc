use crate::d_segment::{find_best_d_segment};
use crate::genes_imgt::{Organism, TcrChain, IMGT_REF};

fn reverse(x:&str) -> String {
    let reversed_str : String = x.chars().rev().collect();
    reversed_str
}

fn get_overlapping_segment(a:&str, b:&str, from_back:bool) -> String{
    if from_back {
        let rev_common : String = a
            .chars()
            .rev()
            .zip(b.chars().rev())
            .take_while(|(a,b)| a==b)
            .map(|x| x.0)
            .collect();
        rev_common.chars().rev().collect()
        
    } else {
        let common: String = a
            .chars()
            .zip(b.chars())
            .take_while(|(a,b)| a==b)
            .map(|(c,_)| c)
            .collect();
        common
    }
}

// MiXCR's V/J alignment scores (KAligner2 defaults)
const MATCH: i32 = 10;
const MISMATCH: i32 = -19;
/// Longest P segment considered after V and before J (repseqio's GermlinePRegionSize).
const MAX_P: usize = 20;

fn complement(b: u8) -> u8 {
    match b {
        b'a' => b't',
        b't' => b'a',
        b'c' => b'g',
        b'g' => b'c',
        other => other,
    }
}

/// Length of the best-scoring ungapped prefix of `query` against `reference` (the first maximum),
/// so an isolated mismatch is crossed when enough matches follow it. None if nothing scores.
fn best_prefix_len(query: impl Iterator<Item = u8>, reference: impl Iterator<Item = u8>) -> Option<usize> {
    let (mut score, mut best, mut best_len) = (0, 0, 0);
    for (i, (q, r)) in query.zip(reference).enumerate() {
        score += if q == r { MATCH } else { MISMATCH };
        if score > best {
            best = score;
            best_len = i + 1;
        }
    }
    (best_len > 0).then_some(best_len)
}

/// Where V ends in the junction. The reference is V followed by its P segment, the reverse
/// complement of V's last bases (repseqio's VRegionWithP), so P-nucleotides count as V.
fn v_end_trimmed(junction: &[u8], v: &[u8]) -> Option<usize> {
    let v_with_p = v.iter().copied().chain(v.iter().rev().take(MAX_P).map(|&b| complement(b)));
    best_prefix_len(junction.iter().copied(), v_with_p)
}

/// Where J begins in the junction, comparing from the end. Read backwards, the reference is J
/// followed by its P segment (the reverse complement of J's first bases).
fn j_begin_trimmed(junction: &[u8], j: &[u8]) -> Option<usize> {
    let j_with_p_rev = j.iter().rev().copied().chain(j.iter().take(MAX_P).map(|&b| complement(b)));
    Some(junction.len() - best_prefix_len(junction.iter().rev().copied(), j_with_p_rev)?)
}

/// Positions in the junction. Trimmed points include P-nucleotides, so they can lie past the
/// germline points; germline points (`v_end`, `d_begin`, `d_end`, `j_begin`) can lie outside the
/// junction. D fields are `None` without a D.
pub struct ReferencePoints {
    pub v_end_trimmed: usize,
    pub d_begin_trimmed: Option<usize>,
    pub d_end_trimmed: Option<usize>,
    pub j_begin_trimmed: usize,
    pub v_end: isize,
    pub d_begin: Option<isize>,
    pub d_end: Option<isize>,
    pub j_begin: isize,
    pub cdr3_end: usize,
    /// AIRR d_call: the D alleles the match fits, sorted, comma-separated
    pub d_call: Option<&'static str>,
}

pub fn determine_reference_points(organism: Organism, junction: &str, v_call: &str, j_call: &str) -> Option<ReferencePoints> {
    let v = IMGT_REF.get(&organism)?.get(v_call)?;
    let j = IMGT_REF.get(&organism)?.get(j_call)?;

    let v_end_trimmed = v_end_trimmed(junction.as_bytes(), v.cdr3_sequence_nt.as_bytes())?;
    let j_begin_trimmed = j_begin_trimmed(junction.as_bytes(), j.cdr3_sequence_nt.as_bytes())?;

    let cdr3_end = junction.len();
    let v_end = v.cdr3_sequence_nt.len() as isize;
    let j_begin = cdr3_end as isize - j.cdr3_sequence_nt.len() as isize;

    // edge case: if trimmed V and J overlap, decide on a middle point
    let (v_end_trimmed, j_begin_trimmed) = if v_end_trimmed > j_begin_trimmed {
        let middle = (v_end_trimmed + j_begin_trimmed) / 2;
        (middle, middle)
    } else {
        (v_end_trimmed, j_begin_trimmed)
    };

    let d = match v.chain {
        TcrChain::TRB => find_best_d_segment(organism, &junction[v_end_trimmed..j_begin_trimmed]),
        _ => None,
    };
    let (d_begin_trimmed, d_end_trimmed, d_begin, d_end, d_call) = match d {
        Some(d) => {
            let begin = v_end_trimmed + d.start;
            let end = begin + d.entry.length as usize;
            // germline D ends: deletions lie outside the match, P-nucleotides inside it
            let germline_begin = begin as isize - d.entry.n_d0 as isize + d.entry.n_p0 as isize;
            let germline_end = end as isize + d.entry.n_d1 as isize - d.entry.n_p1 as isize;
            (Some(begin), Some(end), Some(germline_begin), Some(germline_end), Some(d.d_call))
        }
        None => (None, None, None, None, None),
    };

    Some(ReferencePoints {
        v_end_trimmed,
        d_begin_trimmed,
        d_end_trimmed,
        j_begin_trimmed,
        v_end,
        d_begin,
        d_end,
        j_begin,
        cdr3_end,
        d_call,
    })
}


// fn main() {
// }