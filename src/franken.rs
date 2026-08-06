//! Prototype: generic "frankenmerge candidate generator" for the background/jumbler use case.
//!
//! Pure, published TCRdist-style splice — NO distribution/sampling logic (that stays in the
//! caller). Given a single-chain pool of junctions with their reference points, generate `n`
//! random VALID merges (shared breakpoint exists, no stop codon) and return them as a DataFrame
//! with the distribution key columns (`len_aa`, `n_insertions_vj`). The caller does acceptance.
//!
//! Generation is embarrassingly parallel: the pool is read-only, each merge is independent, so the
//! work is split across rayon threads (each with its own RNG stream) and the per-thread frames are
//! stacked. Shared breakpoints are contiguous INTERVALS, so intersection is a few min/max —
//! allocation-free per attempt.

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use rayon::prelude::*;

// --- tiny splitmix64 PRNG (deterministic; avoids adding the `rand` crate) ---
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    #[inline]
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// --- single-codon translation (standard genetic code), matching junction_trimmer.rs ---
const CODON_AA: &[u8; 64] = b"KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSS*CWCLFLF";
#[inline]
fn nuc_idx(b: u8) -> Option<usize> {
    match b {
        b'A' | b'a' => Some(0),
        b'C' | b'c' => Some(1),
        b'G' | b'g' => Some(2),
        b'T' | b't' => Some(3),
        _ => None,
    }
}
#[inline]
fn translate_codon(c: &[u8]) -> u8 {
    match (nuc_idx(c[0]), nuc_idx(c[1]), nuc_idx(c[2])) {
        (Some(i), Some(j), Some(k)) => CODON_AA[(i << 4) | (j << 2) | k],
        _ => b'X',
    }
}

// Resolve a Python-style (possibly negative) index into [0, len].
#[inline]
fn py_bound(idx: i64, len: usize) -> usize {
    if idx < 0 {
        (len as i64 + idx).max(0) as usize
    } else {
        (idx as usize).min(len)
    }
}

// An inclusive integer interval [lo, hi]; empty if hi < lo.
#[derive(Clone, Copy)]
struct Iv {
    lo: i64,
    hi: i64,
}
impl Iv {
    #[inline]
    fn len(&self) -> i64 {
        (self.hi - self.lo + 1).max(0)
    }
    #[inline]
    fn intersect(&self, o: &Iv) -> Iv {
        Iv { lo: self.lo.max(o.lo), hi: self.hi.min(o.hi) }
    }
}

/// Per-junction precomputed breakpoint intervals (V-anchored positive + J-anchored negative),
/// for before-D and (if D present) after-D.
struct Bkpts {
    before_pos: Iv,
    before_neg: Iv,
    after_pos: Iv, // only meaningful if has_d
    after_neg: Iv,
    has_d: bool,
}

/// Read-only, thread-shareable pool: borrows the (owned, rechunked) column data.
struct Pool<'a> {
    jnt: Vec<&'a [u8]>,
    jaa: Vec<&'a str>,
    vc: Vec<&'a str>,
    jc: Vec<&'a str>,
    ch: Vec<&'a str>,
    vet: Vec<u32>,
    dbt: Vec<u32>,
    det: Vec<u32>,
    jbt: Vec<u32>,
    has_d: Vec<bool>,
    bk: Vec<Bkpts>,
    n: usize,
}

/// Generate `target` valid candidates from `pool` into one DataFrame (single thread / partition).
fn gen_part(pool: &Pool, target: usize, seed: u64, max_attempts: usize) -> PolarsResult<DataFrame> {
    let mut rng = Rng::new(seed);
    let mut out_j: Vec<String> = Vec::with_capacity(target);
    let mut out_aa: Vec<String> = Vec::with_capacity(target);
    let mut out_v: Vec<String> = Vec::with_capacity(target);
    let mut out_jc: Vec<String> = Vec::with_capacity(target);
    let mut out_ch: Vec<String> = Vec::with_capacity(target);
    let mut out_ins: Vec<i32> = Vec::with_capacity(target);
    let mut out_len: Vec<u32> = Vec::with_capacity(target);

    let pool_n = pool.n;
    let mut attempts = 0usize;
    while out_j.len() < target && attempts < max_attempts {
        attempts += 1;
        let a = rng.below(pool_n);
        let mut b = rng.below(pool_n);
        if b == a {
            b = (b + 1) % pool_n;
        }

        let both_d = pool.bk[a].has_d && pool.bk[b].has_d;
        let ivs = [
            (pool.bk[a].before_pos.intersect(&pool.bk[b].before_pos), false),
            (pool.bk[a].before_neg.intersect(&pool.bk[b].before_neg), false),
            (if both_d { pool.bk[a].after_pos.intersect(&pool.bk[b].after_pos) } else { Iv { lo: 1, hi: 0 } }, true),
            (if both_d { pool.bk[a].after_neg.intersect(&pool.bk[b].after_neg) } else { Iv { lo: 1, hi: 0 } }, true),
        ];
        let total: i64 = ivs.iter().map(|(iv, _)| iv.len()).sum();
        if total == 0 {
            continue;
        }
        let mut r = rng.below(total as usize) as i64;
        let mut bkpt = 0i64;
        let mut after_d = false;
        for (iv, is_after) in ivs.iter() {
            let len = iv.len();
            if r < len {
                bkpt = iv.lo + r;
                after_d = *is_after;
                break;
            }
            r -= len;
        }

        // splice nt (Python slice semantics for negative bkpt)
        let (j1, j2) = (pool.jnt[a], pool.jnt[b]);
        let (l1, l2) = (j1.len(), j2.len());
        let cut1 = py_bound(bkpt, l1);
        let cut2 = py_bound(bkpt, l2);
        let mut merged: Vec<u8> = Vec::with_capacity(cut1 + (l2 - cut2));
        merged.extend_from_slice(&j1[..cut1]);
        merged.extend_from_slice(&j2[cut2..]);
        let mlen = merged.len();

        // recompute aa: j1_aa[:aa_bkpt] + translate(boundary codon) + j2_aa[aa_bkpt+1:]
        let aa_bkpt = bkpt.div_euclid(3);
        let a1 = pool.jaa[a].as_bytes();
        let a2 = pool.jaa[b].as_bytes();
        let end1 = py_bound(aa_bkpt, a1.len());
        let start2 = py_bound(aa_bkpt + 1, a2.len());
        let cs = py_bound(aa_bkpt * 3, mlen);
        let ce = py_bound(aa_bkpt * 3 + 3, mlen);
        let boundary = if ce - cs == 3 { translate_codon(&merged[cs..ce]) } else { b'X' };

        let mut aa: Vec<u8> = Vec::with_capacity(end1 + 1 + (a2.len() - start2));
        aa.extend_from_slice(&a1[..end1]);
        aa.push(boundary);
        aa.extend_from_slice(&a2[start2..]);

        if aa.iter().any(|&c| c == b'*') {
            continue;
        }

        // merged reference points -> n_insertions_vj (matches Junction.frankenmerge)
        let j2_offset = if bkpt < 0 { l1 as i64 - l2 as i64 } else { 0 };
        let ve_m = pool.vet[a] as i64;
        let dbt_m = if after_d { pool.dbt[a] as i64 } else { pool.dbt[b] as i64 + j2_offset };
        let det_m = if after_d { pool.det[a] as i64 } else { pool.det[b] as i64 + j2_offset };
        let jbt_m = pool.jbt[b] as i64 + j2_offset;
        let has_d_m = pool.has_d[a] || pool.has_d[b];
        let n_ins = if has_d_m { (dbt_m - ve_m) + (jbt_m - det_m) } else { jbt_m - ve_m };

        out_j.push(String::from_utf8_lossy(&merged).into_owned());
        out_len.push(aa.len() as u32);
        out_aa.push(String::from_utf8_lossy(&aa).into_owned());
        out_v.push(pool.vc[a].to_string());
        out_jc.push(pool.jc[b].to_string());
        out_ch.push(pool.ch[a].to_string());
        out_ins.push(n_ins as i32);
    }

    df![
        "junction" => out_j,
        "junction_aa" => out_aa,
        "v_call" => out_v,
        "j_call" => out_jc,
        "chain" => out_ch,
        "n_insertions_vj" => out_ins,
        "len_aa" => out_len,
    ]
}

#[pyfunction]
#[pyo3(signature = (pool, n, seed=0, max_attempts_factor=10, n_threads=0))]
pub fn franken_candidates(
    py: Python,
    pool: PyDataFrame,
    n: usize,
    seed: u64,
    max_attempts_factor: usize,
    n_threads: usize,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = pool.into();
    let err = |e: PolarsError| pyo3::exceptions::PyValueError::new_err(e.to_string());

    let junction = df.column("junction").map_err(err)?.str().map_err(err)?.rechunk();
    let junction_aa = df.column("junction_aa").map_err(err)?.str().map_err(err)?.rechunk();
    let v_call = df.column("v_call").map_err(err)?.str().map_err(err)?.rechunk();
    let j_call = df.column("j_call").map_err(err)?.str().map_err(err)?.rechunk();
    let chain = df.column("chain").map_err(err)?.str().map_err(err)?.rechunk();

    let col_u32 = |name: &str| -> PyResult<Vec<u32>> {
        Ok(df.column(name).map_err(err)?.u32().map_err(err)?
            .into_iter().map(|o| o.unwrap_or(0)).collect())
    };
    let vet = col_u32("v_end_trimmed")?;
    let dbt = col_u32("d_begin_trimmed")?;
    let det = col_u32("d_end_trimmed")?;
    let jbt = col_u32("j_begin_trimmed")?;
    let has_d: Vec<bool> = df.column("has_d_segment").map_err(err)?.bool().map_err(err)?
        .into_iter().map(|o| o.unwrap_or(false)).collect();

    let jnt: Vec<&[u8]> = junction.into_iter().map(|o| o.unwrap_or("").as_bytes()).collect();
    let jaa: Vec<&str> = junction_aa.into_iter().map(|o| o.unwrap_or("")).collect();
    let vc: Vec<&str> = v_call.into_iter().map(|o| o.unwrap_or("")).collect();
    let jc: Vec<&str> = j_call.into_iter().map(|o| o.unwrap_or("")).collect();
    let ch: Vec<&str> = chain.into_iter().map(|o| o.unwrap_or("")).collect();

    let pool_n = jnt.len();
    if pool_n < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err("pool needs >= 2 junctions"));
    }

    let bk: Vec<Bkpts> = (0..pool_n)
        .map(|i| {
            let l = jnt[i].len() as i64;
            let ve = vet[i] as i64;
            let (before_hi, hasd) = if has_d[i] { (dbt[i] as i64, true) } else { (jbt[i] as i64, false) };
            Bkpts {
                before_pos: Iv { lo: ve, hi: before_hi },
                before_neg: Iv { lo: -l + ve, hi: -l + before_hi },
                after_pos: Iv { lo: det[i] as i64, hi: jbt[i] as i64 },
                after_neg: Iv { lo: -l + det[i] as i64, hi: -l + jbt[i] as i64 },
                has_d: hasd,
            }
        })
        .collect();

    let pool = Pool { jnt, jaa, vc, jc, ch, vet, dbt, det, jbt, has_d, bk, n: pool_n };

    let threads = if n_threads == 0 { rayon::current_num_threads().max(1) } else { n_threads };
    let base = n / threads;
    let rem = n % threads;

    // Parallel generation: one partition per thread, each with an independent RNG stream.
    // GIL released so other Python threads can run while the pure-Rust work goes wide.
    let parts: Vec<PolarsResult<DataFrame>> = py.allow_threads(|| {
        (0..threads)
            .into_par_iter()
            .map(|ti| {
                let target = base + if ti < rem { 1 } else { 0 };
                if target == 0 {
                    return df![
                        "junction" => Vec::<String>::new(), "junction_aa" => Vec::<String>::new(),
                        "v_call" => Vec::<String>::new(), "j_call" => Vec::<String>::new(),
                        "chain" => Vec::<String>::new(), "n_insertions_vj" => Vec::<i32>::new(),
                        "len_aa" => Vec::<u32>::new()
                    ];
                }
                let ma = max_attempts_factor.saturating_mul(target).max(target + 16);
                let s = seed
                    ^ (ti as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(0xD1B5_4A32_D192_ED03);
                gen_part(&pool, target, s, ma)
            })
            .collect()
    });

    // Stack the per-thread frames (cheap: appends chunks).
    let mut frames: Vec<DataFrame> = Vec::with_capacity(threads);
    for p in parts {
        frames.push(p.map_err(err)?);
    }
    let mut out = frames.remove(0);
    for f in frames {
        out.vstack_mut(&f).map_err(err)?;
    }
    Ok(PyDataFrame(out))
}
