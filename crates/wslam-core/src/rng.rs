//! Seeded randomness. There is no other kind in this codebase.
//!
//! spec.md §6: *"Every RNG is seeded and the seed is logged. RANSAC included."*
//! [`DeterministicRng`] has no `from_entropy` constructor, so a caller cannot
//! accidentally introduce non-reproducibility; the seed is logged at
//! construction so a failing CI run reports the seed that produced it.

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// The one randomness source in web-slam.
///
/// ChaCha8 rather than a small PRNG because it is reproducible across
/// architectures — the same seed gives the same stream on aarch64, x86_64 and
/// wasm32, which is a requirement for "native replay and browser replay must
/// agree" (spec.md §6 L3).
#[derive(Debug, Clone)]
pub struct DeterministicRng {
    inner: ChaCha8Rng,
    seed: u64,
    label: &'static str,
}

impl DeterministicRng {
    /// Create a generator, logging the seed.
    ///
    /// `label` identifies the consumer ("ransac-pnp", "vocab-kmeans", ...) so
    /// that a log line is actionable rather than just a number.
    #[must_use]
    pub fn new(label: &'static str, seed: u64) -> Self {
        log::debug!("rng[{label}] seed={seed}");
        DeterministicRng {
            inner: ChaCha8Rng::seed_from_u64(seed),
            seed,
            label,
        }
    }

    /// Derive an independent child stream. Two children with different
    /// `stream_id`s never overlap, so parallel consumers stay reproducible
    /// regardless of scheduling order.
    #[must_use]
    pub fn fork(&self, label: &'static str, stream_id: u64) -> Self {
        // Mix with a 64-bit odd constant so nearby stream ids diverge.
        let seed = self.seed.rotate_left(17) ^ stream_id.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        DeterministicRng::new(label, seed)
    }

    /// The seed this generator was constructed with. Log it with failures.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The consumer label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        self.label
    }

    /// Uniform in `[0, 1)`.
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        self.inner.random::<f64>()
    }

    /// Uniform in `[lo, hi)`.
    #[inline]
    pub fn uniform_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    /// Uniform integer in `[0, n)`. Returns 0 when `n == 0`.
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            self.inner.random_range(0..n)
        }
    }

    /// Standard normal, via Box-Muller. Used to synthesise noise in Tier-1
    /// tests and to sample process noise where a model calls for it.
    pub fn normal(&mut self) -> f64 {
        // Guard against log(0).
        let u1 = self.uniform().max(f64::MIN_POSITIVE);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Normal with given mean and standard deviation.
    #[inline]
    pub fn normal_with(&mut self, mean: f64, stddev: f64) -> f64 {
        mean + stddev * self.normal()
    }

    /// Sample `k` **distinct** indices from `[0, n)` without replacement.
    ///
    /// This is the RANSAC minimal-set primitive. Returns fewer than `k` entries
    /// only when `n < k`. Partial Fisher-Yates over a scratch buffer, so cost is
    /// O(k) after the first call rather than O(n) per sample.
    pub fn sample_distinct(&mut self, n: usize, k: usize, out: &mut Vec<usize>) {
        out.clear();
        if n == 0 {
            return;
        }
        let k = k.min(n);
        // For small k relative to n, rejection sampling is cheaper than
        // building a permutation of n elements every iteration.
        if k * 4 < n {
            while out.len() < k {
                let candidate = self.below(n);
                if !out.contains(&candidate) {
                    out.push(candidate);
                }
            }
        } else {
            let mut pool: Vec<usize> = (0..n).collect();
            for i in 0..k {
                let j = i + self.below(n - i);
                pool.swap(i, j);
                out.push(pool[i]);
            }
        }
    }

    /// Shuffle a slice in place.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            slice.swap(i, self.below(i + 1));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let mut a = DeterministicRng::new("t", 42);
        let mut b = DeterministicRng::new("t", 42);
        for _ in 0..64 {
            assert_eq!(a.uniform().to_bits(), b.uniform().to_bits());
        }
    }

    #[test]
    fn different_seed_different_stream() {
        let mut a = DeterministicRng::new("t", 1);
        let mut b = DeterministicRng::new("t", 2);
        assert_ne!(a.uniform(), b.uniform());
    }

    #[test]
    fn forks_are_independent_and_reproducible() {
        let parent = DeterministicRng::new("p", 7);
        let mut c0 = parent.fork("c", 0);
        let mut c1 = parent.fork("c", 1);
        assert_ne!(c0.uniform(), c1.uniform());

        let mut c0_again = DeterministicRng::new("p", 7).fork("c", 0);
        let mut c0_fresh = parent.fork("c", 0);
        assert_eq!(c0_again.uniform(), c0_fresh.uniform());
    }

    #[test]
    fn sample_distinct_is_distinct_in_both_branches() {
        let mut rng = DeterministicRng::new("t", 9);
        let mut out = Vec::new();
        // rejection branch (k*4 < n)
        rng.sample_distinct(1000, 4, &mut out);
        assert_eq!(out.len(), 4);
        out.sort_unstable();
        out.dedup();
        assert_eq!(out.len(), 4);
        // permutation branch
        rng.sample_distinct(6, 6, &mut out);
        assert_eq!(out.len(), 6);
        out.sort_unstable();
        assert_eq!(out, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn sample_distinct_clamps_to_population() {
        let mut rng = DeterministicRng::new("t", 3);
        let mut out = Vec::new();
        rng.sample_distinct(2, 10, &mut out);
        assert_eq!(out.len(), 2);
        rng.sample_distinct(0, 3, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn normal_is_roughly_standard() {
        let mut rng = DeterministicRng::new("t", 11);
        let n = 20_000;
        let xs: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.03, "mean {mean}");
        assert!((var - 1.0).abs() < 0.05, "var {var}");
    }
}
