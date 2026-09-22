//! Structural filter: a name is retained only if it splits cleanly into
//! (onset)(nucleus)(coda) syllables.
//!
//! The Markov model knows that a letter sequence "looks like" the corpus, but
//! allows unpronounceable consonant clusters if the corpus contains any trace of
//! them. This segmentation enforces pronounceability by construction: if there
//! is no valid segmentation, the name is rejected regardless of its score.

/// Onsets: a single consonant or a cluster attested at the start of a syllable.
/// `qu` consumes both letters; the nucleus follows (qui-ver).
const ONSETS: &[&[u8]] = &[
    b"b", b"c", b"d", b"f", b"g", b"h", b"j", b"k", b"l", b"m", b"n", b"p", b"r", b"s", b"t", b"v",
    b"w", b"x", b"y", b"z", b"bl", b"br", b"ch", b"cl", b"cr", b"dr", b"fl", b"fr", b"gl", b"gr",
    b"kl", b"kr", b"ph", b"pl", b"pr", b"qu", b"sc", b"sh", b"sk", b"sl", b"sm", b"sn", b"sp",
    b"st", b"sw", b"th", b"tr", b"tw", b"vr", b"wh", b"wr", b"scr", b"shr", b"spl", b"spr", b"str",
    b"thr",
];

/// Nuclei: a single vowel or a common vowel digraph.
const NUCLEI: &[&[u8]] = &[
    b"a", b"e", b"i", b"o", b"u", b"y", b"ai", b"au", b"aw", b"ay", b"ea", b"ee", b"ei", b"eo",
    b"eu", b"ew", b"ey", b"ia", b"ie", b"io", b"iu", b"oa", b"oe", b"oi", b"oo", b"ou", b"ow",
    b"oy", b"ua", b"ue", b"ui", b"uo", b"ya", b"ye", b"yo", b"yu",
];

/// Codas: empty, a single consonant, or a cluster attested at a syllable ending.
const CODAS: &[&[u8]] = &[
    b"", b"b", b"c", b"d", b"f", b"g", b"k", b"l", b"m", b"n", b"p", b"r", b"s", b"t", b"v", b"x",
    b"z", b"ck", b"ct", b"ff", b"ft", b"gh", b"ld", b"lf", b"lk", b"lm", b"lp", b"lt", b"ll",
    b"mb", b"mp", b"nd", b"ng", b"nk", b"ns", b"nt", b"pt", b"rb", b"rd", b"rk", b"rl", b"rm",
    b"rn", b"rp", b"rs", b"rt", b"sh", b"sk", b"sp", b"ss", b"st", b"th", b"tt", b"zz",
];

const MAX_SYL: usize = 4;

/// Returns true when `w` has at least one segmentation into
/// `min_syl..=max_syl` syllables.
pub fn is_pronounceable(w: &[u8], min_syl: usize, max_syl: usize) -> bool {
    let n = w.len();
    // reach[p]: bit mask of the syllable counts that can reach position p. A
    // simple boolean is insufficient because the final count must be constrained.
    let mut reach = vec![0u8; n + 1];
    reach[0] = 1; // zero syllables consumed at position 0

    for pos in 0..n {
        if reach[pos] == 0 {
            continue;
        }
        let counts = reach[pos];

        for onset in ONSETS.iter().copied().chain(
            // An empty onset is allowed only at the start of a word; elsewhere
            // it would create a hard-to-pronounce hiatus between two nuclei.
            std::iter::once(&b""[..]).filter(|_| pos == 0),
        ) {
            let a = pos + onset.len();
            if a > n || &w[pos..a] != onset {
                continue;
            }
            for nucleus in NUCLEI {
                let b = a + nucleus.len();
                if b > n || &w[a..b] != *nucleus {
                    continue;
                }
                for coda in CODAS {
                    let c = b + coda.len();
                    if c > n || &w[b..c] != *coda {
                        continue;
                    }
                    // A non-empty internal coda may join the following onset;
                    // leave that decision to the dynamic program.
                    reach[c] |= counts << 1;
                }
            }
        }
    }

    let wanted: u8 = (min_syl..=max_syl.min(MAX_SYL)).fold(0, |m, s| m | (1 << s));
    reach[n] & wanted != 0
}

/// Near-zero-cost rejections applied before segmentation.
pub fn plausible(w: &[u8]) -> bool {
    let mut vowels = 0;
    let mut run_consonant = 0;
    let mut prev = 0u8;
    let mut repeat = 1;

    for &b in w {
        let is_vowel = matches!(b, b'a' | b'e' | b'i' | b'o' | b'u' | b'y');
        if is_vowel {
            vowels += 1;
            run_consonant = 0;
        } else {
            run_consonant += 1;
            if run_consonant > 3 {
                return false;
            }
        }
        if b == prev {
            repeat += 1;
            if repeat > 2 {
                return false; // no letter may appear three times in a row
            }
        } else {
            repeat = 1;
        }
        prev = b;
    }

    (2..=4).contains(&vowels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_a_simple_segmentation() {
        assert!(is_pronounceable(b"voronoi", 3, 4));
    }

    #[test]
    fn long_words_do_not_overflow_the_dynamic_program() {
        assert!(!is_pronounceable(b"bcdfghjklmnpqrstvwxyz", 1, 4));
    }
}
