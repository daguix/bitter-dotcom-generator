//! Bounded edit distance used to reject names that are too close to a seed.
//!
//! The n-gram filter misses cases like `cenroid` and `centroid`: they share no
//! 5-gram even though they differ by only one letter. A technical term that is
//! one letter off does not read like an invented name to engineers who know the
//! real word; it reads like a typo.

/// Returns true when the Levenshtein distance between `a` and `b` is strictly
/// less than `max`. Stops as soon as the entire current row reaches `max`, which
/// is much faster than a full calculation when the words differ.
pub fn closer_than(a: &[u8], b: &[u8], max: usize) -> bool {
    if a.len().abs_diff(b.len()) >= max {
        return false;
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];

    for (i, &ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        let mut row_min = cur[0];
        for (j, &cb) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(ca != cb);
            cur[j + 1] = sub.min(prev[j + 1] + 1).min(cur[j] + 1);
            row_min = row_min.min(cur[j + 1]);
        }
        // The entire row is already beyond the threshold; subsequent rows can
        // only increase, so the result is known.
        if row_min >= max {
            return false;
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()] < max
}

/// Returns true when `word` is fewer than `max` edits away from any reference.
pub fn near_any(word: &str, refs: &[String], max: usize) -> bool {
    let m = word.as_bytes();
    refs.iter().any(|r| closer_than(m, r.as_bytes(), max))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_distances() {
        assert!(closer_than(b"cenroid", b"centroid", 2)); // one insertion
        assert!(!closer_than(b"cenroid", b"centroid", 1));
        assert!(closer_than(b"fraccal", b"fractal", 2));
        assert!(!closer_than(b"nodagon", b"decagon", 3)); // three edits
        assert!(closer_than(b"multope", b"multus", 4));
        assert!(!closer_than(b"multope", b"multus", 3));
    }

    #[test]
    fn length_difference_short_circuits() {
        assert!(!closer_than(b"ab", b"abcdef", 3));
    }
}
