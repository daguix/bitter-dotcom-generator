//! Exhaustive traversal of the [a-z]^L space.
//!
//! 26^6 + 26^7 ≈ 8.34 billion strings: this is enumerable as long as we do not
//! compute a complete score for every string. Two ideas make this possible:
//!
//! 1. **Incremental scoring** — descend the prefix tree while accumulating log
//!    probability. Each leaf costs only one more table lookup than its parent.
//! 2. **Pruning** — all log probabilities are negative, so the cumulative score
//!    can only decrease. Once a prefix falls below the threshold, no suffix can
//!    recover, and the entire branch can be cut.
//!
//! Most of the search space dies within the first three letters.

use crate::markov::{BOUND, Markov};
use crate::syllable;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct Candidate {
    pub name: String,
    pub score: f32,
}

pub struct ScanOpts {
    pub len: usize,
    pub threshold: f32,
    pub min_syl: usize,
    pub max_syl: usize,
}

pub fn scan(m: &Markov, o: &ScanOpts) -> Vec<Candidate> {
    assert!(o.len >= 3, "length is too short to initialize the context");
    // The threshold applies to the average per transition. Denormalize it once
    // so it can be compared directly with the cumulative score while descending.
    let floor = o.threshold * (o.len + 1) as f32;

    // Parallelize over the first two letters: 676 tasks, each large compared
    // with scheduling overhead and numerous enough to balance the workload.
    (0..26u16 * 26)
        .into_par_iter()
        .flat_map_iter(|prefix| {
            let (c0, c1) = ((prefix / 26) as u8, (prefix % 26) as u8);
            let mut buf = vec![0u8; o.len];
            let mut out = Vec::new();

            let mut ctx = Markov::ctx(BOUND, BOUND, BOUND);
            let mut acc = m.logp(ctx, c0);
            ctx = Markov::advance(ctx, c0);
            acc += m.logp(ctx, c1);
            ctx = Markov::advance(ctx, c1);
            buf[0] = c0;
            buf[1] = c1;

            if acc >= floor {
                descend(m, o, floor, &mut buf, 2, ctx, acc, &mut out);
            }
            out.into_iter()
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn descend(
    m: &Markov,
    o: &ScanOpts,
    floor: f32,
    buf: &mut [u8],
    pos: usize,
    ctx: usize,
    acc: f32,
    out: &mut Vec<Candidate>,
) {
    if pos == o.len {
        let total = acc + m.logp(ctx, BOUND);
        if total < floor {
            return;
        }
        // Structural filters apply only here: they cost much more than a table
        // lookup, and the threshold has already rejected the vast majority.
        let word: Vec<u8> = buf.iter().map(|&s| s + b'a').collect();
        if !syllable::plausible(&word) || !syllable::is_pronounceable(&word, o.min_syl, o.max_syl) {
            return;
        }
        out.push(Candidate {
            name: String::from_utf8(word).expect("ASCII by construction"),
            score: total / (o.len + 1) as f32,
        });
        return;
    }

    for s in 0..26u8 {
        let next = acc + m.logp(ctx, s);
        if next < floor {
            continue; // dead branch: the cumulative score cannot recover
        }
        buf[pos] = s;
        descend(
            m,
            o,
            floor,
            buf,
            pos + 1,
            Markov::advance(ctx, s),
            next,
            out,
        );
    }
}
