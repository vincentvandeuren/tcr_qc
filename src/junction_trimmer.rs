const CODON_AA: &[u8; 64] =
    b"KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSS*CWCLFLF";

#[inline(always)]
fn nuc_idx(b: u8) -> Option<usize> {
    match b {
        b'A' | b'a' => Some(0),
        b'C' | b'c' => Some(1),
        b'G' | b'g' => Some(2),
        b'T' | b't' => Some(3),
        _ => None,
    }
}

#[inline(always)]
fn translate_codon(a: u8, b: u8, c: u8) -> Option<u8> {
    let i = nuc_idx(a)?;
    let j = nuc_idx(b)?;
    let k = nuc_idx(c)?;
    Some(CODON_AA[(i << 4) | (j << 2) | k])
}

/// Translate `seq` in frame 0, appending the residues to `buf`. A trailing 1-2 nt remainder is
/// dropped; a codon containing anything but ACGT/acgt becomes `X`; stop codons become `*`.
pub fn translate_into(seq: &[u8], buf: &mut String) {
    buf.extend(
        seq.chunks_exact(3)
            .map(|c| translate_codon(c[0], c[1], c[2]).unwrap_or(b'X') as char),
    );
}

fn trim_nt<'a>(nt: &'a [u8], aa: &[u8]) -> Option<&'a [u8]> {
    let coding_len = aa.len() * 3;
    if nt.len() < coding_len {
        return None;
    }
    'outer: for start in 0..=(nt.len() - coding_len) {
        for (i, &residue) in aa.iter().enumerate() {
            let j = start + i * 3;
            match translate_codon(nt[j], nt[j + 1], nt[j + 2]) {
                Some(translated) if translated == residue => {}
                _ => continue 'outer,
            }
        }
        return Some(&nt[start..start + coding_len]);
    }
    None
}

pub fn trim_junction_to_cdr3(junction: &str, junction_aa: &str) -> Option<String> {
    let coding_len = junction_aa.len() * 3;

    if junction.len() == coding_len {
        return Some(junction.to_lowercase());
    }

    if junction.len() < coding_len {
        return None;
    }

    let trimmed = trim_nt(junction.as_bytes(), junction_aa.as_bytes())?;
    Some(std::str::from_utf8(trimmed).unwrap().to_lowercase())
}