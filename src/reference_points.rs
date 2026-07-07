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

fn get_overlap_index(a: &str, b: &str, from_back: bool) -> Option<usize> {
    let overlap_len = if from_back {
        a.chars()
            .rev()
            .zip(b.chars().rev())
            .take_while(|(a, b)| a == b)
            .count()
    } else {
        a.chars()
            .zip(b.chars())
            .take_while(|(a, b)| a == b)
            .count()
    };

    if overlap_len == 0 {
        None
    } else if from_back {
        // index of where the trailing overlap starts
        Some(a.chars().count() - overlap_len)
    } else {
        // index of the first non-matching char from the front
        Some(overlap_len)
    }
}

pub struct ReferencePoints {
    pub v_end_trimmed: usize,
    pub d_begin_trimmed: usize,
    pub d_end_trimmed: usize,
    pub j_begin_trimmed: usize,
    pub v_end: usize,
    pub d_begin: usize,
    pub d_end: usize,
    pub j_begin: usize,
    pub cdr3_end: usize,
}

pub fn determine_reference_points(organism: Organism, junction: &str, v_call: &str, j_call: &str) -> Option<ReferencePoints> {
    let v = IMGT_REF.get(&organism)?.get(v_call)?;
    let j = IMGT_REF.get(&organism)?.get(j_call)?;

    // println!("V call: {}, J call: {}", v.cdr3_sequence_nt, j.cdr3_sequence_nt);

    let v_end_trimmed = get_overlap_index(junction, v.cdr3_sequence_nt.as_str(), false)?;
    let j_begin_trimmed = get_overlap_index(junction, j.cdr3_sequence_nt.as_str(), true)?;

    // println!("v_end_trimmed: {}, j_begin_trimmed: {}", v_end_trimmed, j_begin_trimmed);

    let cdr3_end = junction.chars().count();
    // let j_begin = cdr3_end - j.cdr3_nucseq_len;
    let j_begin = cdr3_end.saturating_sub(j.cdr3_sequence_nt.chars().count());


    // edge case: if trimmed V and J overlap, decide on a middle point
    let (v_end_trimmed, j_begin_trimmed) = if v_end_trimmed > j_begin_trimmed {
        // println!("Warning: V and J trimmed positions overlap. This may indicate an unusual recombination event or a potential issue with the reference data. Resolving by taking the middle point of the overlap.");
        let middle = (v_end_trimmed + j_begin_trimmed) / 2;

        (middle, middle)
    } else {

        (v_end_trimmed, j_begin_trimmed)
    };
    // println!("After resolving V-J overlap, v_end_trimmed: {}, j_begin_trimmed: {}", v_end_trimmed, j_begin_trimmed);

    let (d_begin_trimmed, d_end_trimmed, d_begin, d_end) = match v.chain {
        TcrChain::TRB => {
            // println!("Processing D for TRB chain");
            let vj_junction = &junction[v_end_trimmed..j_begin_trimmed];
            // println!("V-J junction sequence: {}", vj_junction);
            match find_best_d_segment(organism, vj_junction) {
                Some(best_d) => {
                    // println!("Best matching D segment: {}", best_d.substring);
                    let d_begin_trimmed = v_end_trimmed + vj_junction.find(&best_d.substring)?;
                    let d_end_trimmed = d_begin_trimmed + best_d.length as usize;
                    let d_begin = d_begin_trimmed.saturating_sub(best_d.n_d0 as usize);
                    let d_end = d_end_trimmed + best_d.n_d1 as usize;
                    (d_begin_trimmed, d_end_trimmed, d_begin, d_end)
                }
                None => {
                    // println!("No D segment found");
                    (0, 0, 0, 0)
                }
            } 
        }
        _ => (0, 0, 0, 0),
    };

    Some(ReferencePoints {
        v_end_trimmed: v_end_trimmed,
        j_begin_trimmed: j_begin_trimmed,
        d_begin_trimmed: d_begin_trimmed,
        d_end_trimmed: d_end_trimmed,
        v_end: v.cdr3_sequence_nt.chars().count(),
        d_begin: d_begin,
        d_end: d_end,
        j_begin: j_begin,
        cdr3_end: cdr3_end,
    })
}



// fn main() {
// }