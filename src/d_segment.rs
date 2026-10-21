use std::collections::{hash_map::Entry, HashMap};
use std::sync::LazyLock;
use aho_corasick::{AhoCorasick, AhoCorasickBuilder};

use crate::genes_imgt::{IMGT_HUMAN_TRBD, IMGT_MOUSE_TRBD, Organism};

/// Shortest D match accepted: MiXCR requires a D alignment score of 25 at +5 per match.
const MIN_D_LEN: usize = 5;

/// One unique substring of a D allele extended with its P segments, rc(D) + D + rc(D), at the
/// placement that explains it with the fewest deletions, then the fewest P-nucleotides.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DGeneSubstring {
    pub substring: String,
    pub length: u16,
    /// D bases deleted at the 5' and 3' end (0 at an end with P-nucleotides)
    pub n_d0: u16,
    pub n_d1: u16,
    /// P-nucleotides at the 5' and 3' end
    pub n_p0: u16,
    pub n_p1: u16,
    /// Bit i is set when the substring occurs in allele i of the organism's D list
    pub alleles: u32,
}

impl DGeneSubstring {
    /// Lower is better between matches of equal length: fewest deletions, then fewest P.
    fn rank(&self) -> (u16, u16) {
        (self.n_d0 + self.n_d1, self.n_p0 + self.n_p1)
    }

    /// Order between placements of the same substring.
    fn placement_key(&self) -> (u16, u16, u16, u16) {
        let (deletions, p) = self.rank();
        (deletions, p, self.n_d0, self.n_d1)
    }
}

fn reverse_complement(s: &str) -> String {
    s.bytes()
        .rev()
        .map(|b| match b {
            b'a' => 't',
            b't' => 'a',
            b'c' => 'g',
            b'g' => 'c',
            other => other as char,
        })
        .collect()
}

/// All substrings of rc(D) + D + rc(D) of at least MIN_D_LEN that overlap D itself, so that
/// P-nucleotides can extend an untrimmed end but a match inside a flank (an inverted D) is not
/// indexed. A match may hold at most as many P-nucleotides as germline D bases. A substring
/// found at several placements or in several alleles keeps the best placement and the union of
/// the alleles.
fn build_unique_substrings(alleles: &[(&str, &str)]) -> Vec<DGeneSubstring> {
    assert!(alleles.len() <= 32, "allele mask holds at most 32 D alleles");
    let mut best_by_sub: HashMap<String, DGeneSubstring> = HashMap::new();

    for (bit, (_, d)) in alleles.iter().enumerate() {
        let l = d.len();
        let rc = reverse_complement(d);
        let extended = format!("{rc}{d}{rc}");
        for start in 0..2 * l {
            for end in (start + MIN_D_LEN).max(l + 1)..=3 * l {
                let p = l.saturating_sub(start) + end.saturating_sub(2 * l);
                let germline = end.min(2 * l) - start.max(l);
                if p > germline {
                    continue;
                }
                let cand = DGeneSubstring {
                    substring: extended[start..end].to_string(),
                    length: (end - start) as u16,
                    n_d0: start.saturating_sub(l) as u16,
                    n_d1: (2 * l).saturating_sub(end) as u16,
                    n_p0: l.saturating_sub(start) as u16,
                    n_p1: end.saturating_sub(2 * l) as u16,
                    alleles: 1 << bit,
                };
                match best_by_sub.entry(cand.substring.clone()) {
                    Entry::Vacant(e) => {
                        e.insert(cand);
                    }
                    Entry::Occupied(mut e) => {
                        let existing = e.get_mut();
                        let alleles = existing.alleles | cand.alleles;
                        if cand.placement_key() < existing.placement_key() {
                            *existing = cand;
                        }
                        existing.alleles = alleles;
                    }
                }
            }
        }
    }

    // Deterministic order (not required for AC, but nice for tests/debugging).
    let mut entries: Vec<_> = best_by_sub.into_values().collect();
    entries.sort_unstable_by(|a, b| {
        a.length
            .cmp(&b.length)
            .then_with(|| a.substring.cmp(&b.substring))
    });
    entries
}

struct AcIndex {
    ac: AhoCorasick,
    entries: Vec<DGeneSubstring>, // pattern i corresponds to entries[i]
    /// AIRR d_call for every allele mask: the names of the set bits, comma-separated
    calls: Vec<String>,
}

fn build_ac_index(alleles: &[(&str, &str)]) -> AcIndex {
    let entries = build_unique_substrings(alleles);
    let patterns = entries.iter().map(|e| e.substring.as_str());

    let ac = AhoCorasickBuilder::new()
        .ascii_case_insensitive(false)
        .build(patterns)
        .expect("failed to build Aho-Corasick automaton");

    let calls = (0..1u32 << alleles.len())
        .map(|mask| {
            alleles
                .iter()
                .enumerate()
                .filter(|(bit, _)| mask >> bit & 1 == 1)
                .map(|(_, (name, _))| *name)
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect();

    AcIndex { ac, entries, calls }
}


static HUMAN_IDX: LazyLock<AcIndex> = LazyLock::new(|| build_ac_index(&IMGT_HUMAN_TRBD));
static MOUSE_IDX: LazyLock<AcIndex> = LazyLock::new(|| build_ac_index(&IMGT_MOUSE_TRBD));

pub struct DMatch {
    pub entry: &'static DGeneSubstring,
    /// Start of the match in the searched window
    pub start: usize,
    /// AIRR d_call: every allele that explains an equally good match, sorted, comma-separated
    pub d_call: &'static str,
}

/// Fast single-pass search using Aho–Corasick:
/// - Longest match
/// - If tied, fewest deletions (n_d0 + n_d1), then fewest P-nucleotides (n_p0 + n_p1)
/// - Matches still tied, at any position, pool their alleles; the first one gives the position
pub fn find_best_d_segment(organism: Organism, non_templated: &str) -> Option<DMatch> {
    let idx: &'static AcIndex = match organism {
        Organism::Human => &*HUMAN_IDX,
        Organism::Mouse => &*MOUSE_IDX,
    };

    let mut best: Option<(&'static DGeneSubstring, usize, u32)> = None;

    for m in idx.ac.find_overlapping_iter(non_templated) {
        let e = &idx.entries[m.pattern().as_usize()];
        match &mut best {
            Some((b, _, alleles)) if e.length == b.length && e.rank() == b.rank() => *alleles |= e.alleles,
            Some((b, _, _)) if e.length < b.length || (e.length == b.length && e.rank() > b.rank()) => {}
            _ => best = Some((e, m.start(), e.alleles)),
        }
    }

    best.map(|(entry, start, alleles)| DMatch { entry, start, d_call: &idx.calls[alleles as usize] })
}
