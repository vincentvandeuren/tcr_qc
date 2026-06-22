use std::collections::{hash_map::Entry, HashMap};
use std::sync::LazyLock;
use aho_corasick::{AhoCorasick, AhoCorasickBuilder};

// use crate::genes::{HUMAN_TRBD_SEQS};
use crate::genes_imgt::{IMGT_HUMAN_TRBD, IMGT_MOUSE_TRBD, Organism};

/// Metadata per unique substring (deduped across all D segments).
/// We store counters as u16 internally (compact) but return u32 to Polars for generality.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DGeneSubstring {
    pub full_seq: String,
    pub substring: String,
    pub n_d0: u16,
    pub n_d1: u16,
    pub length: u16,
}

impl DGeneSubstring {
    // #[inline]
    fn total_deletions(&self) -> u16 {
        self.n_d0.saturating_add(self.n_d1)
    }
}

fn build_unique_substrings<S: AsRef<str>>(
    seqs: impl IntoIterator<Item = S>,
    min_length: usize,
) -> Vec<DGeneSubstring> {
    let mut best_by_sub: HashMap<String, DGeneSubstring> = HashMap::new();

    for seq_s in seqs {
        let seq = seq_s.as_ref();
        let n = seq.len();
        if n < min_length {
            continue;
        }

        for length in min_length..=n {
            for start in 0..=n - length {
                let substring = &seq[start..start + length];
                let cand = DGeneSubstring {
                    full_seq: seq.to_string(),
                    substring: substring.to_string(),
                    n_d0: start as u16,
                    n_d1: (n - (start + length)) as u16,
                    length: length as u16,
                };

                let cand_sum = cand.total_deletions();
                match best_by_sub.entry(cand.substring.clone()) {
                    Entry::Vacant(e) => {
                        e.insert(cand);
                    }
                    Entry::Occupied(mut e) => {
                        let existing = e.get();
                        let exist_sum = existing.total_deletions();
                        if cand_sum < exist_sum
                            || (cand_sum == exist_sum
                                && (cand.n_d0, cand.n_d1, &cand.full_seq, &cand.substring)
                                    < (existing.n_d0, existing.n_d1, &existing.full_seq, &existing.substring))
                        {
                            e.insert(cand);
                        }
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
}

fn build_ac_index<S: AsRef<str>>(seqs: impl IntoIterator<Item = S>, min_length: usize) -> AcIndex {
    let entries = build_unique_substrings(seqs, min_length);
    // println!("Built AC index with {} unique substrings", entries.len());
    let patterns = entries.iter().map(|e| e.substring.as_str());

    let ac = AhoCorasickBuilder::new()
        .ascii_case_insensitive(false)
        .build(patterns)
        .expect("failed to build Aho-Corasick automaton");

    AcIndex { ac, entries }
}


static HUMAN_IDX: LazyLock<AcIndex> = LazyLock::new(|| build_ac_index(IMGT_HUMAN_TRBD.iter(), 3));
static MOUSE_IDX: LazyLock<AcIndex> = LazyLock::new(|| build_ac_index(IMGT_MOUSE_TRBD.iter(), 3));

/// Fast single-pass search using Aho–Corasick:
/// - Longest substring length
/// - If tied, fewest deletions (n_d0 + n_d1)
pub fn find_best_d_segment(organism: Organism, non_templated: &str) -> Option<&'static DGeneSubstring> {
    let idx: &'static AcIndex = match organism {
        Organism::Human => &*HUMAN_IDX,
        Organism::Mouse => &*MOUSE_IDX,
    };

    let mut best_id: Option<usize> = None;
    let mut best_len: u16 = 0;
    let mut best_sum: u16 = u16::MAX;

    for m in idx.ac.find_overlapping_iter(non_templated) {
        let id = m.pattern().as_usize();
        let e = &idx.entries[id];
        let len = e.length;
        let sum = e.total_deletions();

        if len > best_len || (len == best_len && sum < best_sum) {
            best_len = len;
            best_sum = sum;
            best_id = Some(id);
        }
    }

    best_id.map(|i| &idx.entries[i])
}
