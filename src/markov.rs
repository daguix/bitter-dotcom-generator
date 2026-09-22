//! Order-3 character n-gram model with Witten-Bell smoothing.
//!
//! The seed corpus is small (a few hundred words), so many order-3 contexts are
//! never observed. Witten-Bell then backs off to lower orders, weighted by
//! contextual diversity: a context seen with many different continuations is
//! trusted more than one seen only once. There are no hyperparameters to tune.

/// 26 letters plus one boundary symbol (start and end of word).
pub const ALPHA: usize = 27;
pub const BOUND: u8 = 26;

const N1: usize = ALPHA;
const N2: usize = ALPHA * ALPHA;
const N3: usize = ALPHA * ALPHA * ALPHA;

pub struct Markov {
    /// log P(next | 3 preceding symbols), indexed by `ctx3 * ALPHA + next`.
    logp: Vec<f32>,
}

#[derive(Default)]
struct Counts {
    c0: Vec<u32>,
    c1: Vec<u32>,
    c2: Vec<u32>,
    c3: Vec<u32>,
}

impl Markov {
    pub fn train(words: &[String]) -> Markov {
        let mut k = Counts {
            c0: vec![0; ALPHA],
            c1: vec![0; N1 * ALPHA],
            c2: vec![0; N2 * ALPHA],
            c3: vec![0; N3 * ALPHA],
        };

        for w in words {
            // ^^^ word $: three leading boundaries give the first real character
            // a complete order-3 context.
            let mut syms = vec![BOUND; 3];
            syms.extend(w.bytes().map(|b| b - b'a'));
            syms.push(BOUND);

            for i in 3..syms.len() {
                let (a, b, c) = (
                    syms[i - 3] as usize,
                    syms[i - 2] as usize,
                    syms[i - 1] as usize,
                );
                let nxt = syms[i] as usize;
                k.c3[((a * ALPHA + b) * ALPHA + c) * ALPHA + nxt] += 1;
                k.c2[(b * ALPHA + c) * ALPHA + nxt] += 1;
                k.c1[c * ALPHA + nxt] += 1;
                k.c0[nxt] += 1;
            }
        }

        Markov {
            logp: build_table(&k),
        }
    }

    /// Context index for three consecutive symbols.
    #[inline]
    pub fn ctx(a: u8, b: u8, c: u8) -> usize {
        (a as usize * ALPHA + b as usize) * ALPHA + c as usize
    }

    /// Shifts the context by one symbol.
    #[inline]
    pub fn advance(ctx: usize, next: u8) -> usize {
        (ctx % (ALPHA * ALPHA)) * ALPHA + next as usize
    }

    #[inline]
    pub fn logp(&self, ctx: usize, next: u8) -> f32 {
        self.logp[ctx * ALPHA + next as usize]
    }

    /// Average log probability per transition, including the final boundary.
    /// Normalized so seven-letter names are not penalized against six-letter ones.
    pub fn score(&self, word: &str) -> f32 {
        let mut ctx = Markov::ctx(BOUND, BOUND, BOUND);
        let mut total = 0.0;
        let mut n = 0;
        for b in word.bytes() {
            let s = b - b'a';
            total += self.logp(ctx, s);
            ctx = Markov::advance(ctx, s);
            n += 1;
        }
        total += self.logp(ctx, BOUND);
        total / (n + 1) as f32
    }
}

/// Flattens counts into a dense table of smoothed log probabilities.
fn build_table(k: &Counts) -> Vec<f32> {
    let total0: u32 = k.c0.iter().sum();

    // Order 0: add-one smoothing so no symbol has zero probability.
    let p0: Vec<f32> = (0..ALPHA)
        .map(|x| (k.c0[x] + 1) as f32 / (total0 + ALPHA as u32) as f32)
        .collect();

    // p0 is indexed by symbol alone, not by context, hence the constant zero.
    let p1 = backoff_level(&k.c1, &p0, N1, |_| 0);
    let p2 = backoff_level(&k.c2, &p1, N2, |ctx| ctx % N1);
    let p3 = backoff_level(&k.c3, &p2, N3, |ctx| ctx % N2);

    p3.iter().map(|p| p.ln()).collect()
}

/// One Witten-Bell layer: P(x|ctx) = (c + T·P_lower(x)) / (N + T), where N
/// is the observed mass of the context and T its number of distinct continuations.
fn backoff_level(
    counts: &[u32],
    lower: &[f32],
    n_ctx: usize,
    lower_ctx: impl Fn(usize) -> usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; n_ctx * ALPHA];
    for ctx in 0..n_ctx {
        let row = &counts[ctx * ALPHA..(ctx + 1) * ALPHA];
        let n: u32 = row.iter().sum();
        let t = row.iter().filter(|&&c| c > 0).count() as f32;
        let lo = lower_ctx(ctx) * ALPHA;
        let denom = n as f32 + t;

        for x in 0..ALPHA {
            let p_lower = lower[lo + x];
            out[ctx * ALPHA + x] = if n == 0 {
                // Unseen context: delegate entirely to the lower order.
                p_lower
            } else {
                (row[x] as f32 + t * p_lower) / denom
            };
        }
    }
    out
}
