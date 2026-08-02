//! Homography estimation: normalised DLT plus RANSAC over a seeded RNG.
//!
//! L2 fits one homography per frame pair during the init pan (spec.md §4 L2).
//! Two details are load-bearing rather than decorative:
//!
//! 1. **Hartley normalisation.** Raw pixel coordinates on a 1280x720 frame make
//!    the DLT design matrix carry entries spanning `1` to `10^6`, and the
//!    smallest singular vector of a matrix with a condition number that large is
//!    noise. Isotropic normalisation to zero centroid and mean radius `sqrt(2)`
//!    is the standard fix and it is not optional here.
//! 2. **A deliberately loose RANSAC threshold.** RANSAC in this layer exists to
//!    reject mismatched features, *not* to enforce the pinhole model. A tight
//!    threshold would silently discard exactly the peripheral correspondences
//!    that carry the radial-distortion signal the refinement in
//!    [`crate::refine`] needs, which would hide the failure mode spec.md §6 L2
//!    makes a gate.

use nalgebra::DMatrix;
use wslam_core::{DeterministicRng, Mat3, Scalar, Vec2};

/// Minimum correspondences for the eight-degree-of-freedom DLT: each
/// correspondence contributes two rows.
pub const MIN_HOMOGRAPHY_MATCHES: usize = 4;

/// Minimal sample size drawn by RANSAC.
const MINIMAL_SET: usize = 4;

/// Ratio between the second-smallest and largest singular value of the DLT
/// design matrix below which the null space is wider than one dimension.
///
/// Four collinear points, or a set with repeats, leave the homography
/// underdetermined; the SVD still returns *a* vector and it is meaningless.
/// Detecting the rank deficiency here is cheaper than discovering it as a
/// wildly wrong focal length three stages later.
const NULLSPACE_RATIO: Scalar = 1e-9;

/// Target probability that RANSAC has drawn at least one all-inlier sample,
/// used for the adaptive stopping rule.
const RANSAC_CONFIDENCE: Scalar = 0.9999;

/// Apply a homography to a point, returning `None` if the point maps to
/// infinity (`w == 0`), which a caller must not silently treat as a pixel.
#[must_use]
pub fn apply_homography(h: &Mat3, p: Vec2) -> Option<Vec2> {
    let w = h[(2, 0)] * p.x + h[(2, 1)] * p.y + h[(2, 2)];
    if w.abs() < 1e-12 {
        return None;
    }
    Some(Vec2::new(
        (h[(0, 0)] * p.x + h[(0, 1)] * p.y + h[(0, 2)]) / w,
        (h[(1, 0)] * p.x + h[(1, 1)] * p.y + h[(1, 2)]) / w,
    ))
}

/// Sum of squared forward and backward transfer distances, in px^2.
///
/// Symmetric rather than one-way because both images carry detector noise; a
/// one-way error silently privileges whichever frame happens to be the source.
#[must_use]
pub fn symmetric_transfer_error(h: &Mat3, h_inv: &Mat3, p: Vec2, q: Vec2) -> Scalar {
    let forward = match apply_homography(h, p) {
        Some(v) => (v - q).norm_squared(),
        None => Scalar::INFINITY,
    };
    let backward = match apply_homography(h_inv, q) {
        Some(v) => (v - p).norm_squared(),
        None => Scalar::INFINITY,
    };
    forward + backward
}

/// Scale a homography to unit Frobenius norm with a deterministic sign.
///
/// A homography is only defined up to scale, so every consumer that compares
/// two of them needs one canonical representative. The sign is fixed by forcing
/// the largest-magnitude entry positive.
#[must_use]
pub fn normalize_homography(h: &Mat3) -> Mat3 {
    let norm = h.norm();
    if !(norm > 0.0) || !norm.is_finite() {
        return *h;
    }
    let mut out = h / norm;
    let mut extreme: Scalar = 0.0;
    for v in out.iter() {
        if v.abs() > extreme.abs() {
            extreme = *v;
        }
    }
    if extreme < 0.0 {
        out = -out;
    }
    out
}

/// Isotropic similarity taking `pts` to zero centroid and mean radius
/// `sqrt(2)` (Hartley). `None` when every point coincides.
fn normalizing_transform(pts: &[Vec2]) -> Option<Mat3> {
    if pts.is_empty() {
        return None;
    }
    let n = pts.len() as Scalar;
    let centroid = pts.iter().fold(Vec2::zeros(), |acc, p| acc + p) / n;
    let mean_radius = pts.iter().map(|p| (p - centroid).norm()).sum::<Scalar>() / n;
    if !(mean_radius > 1e-12) || !mean_radius.is_finite() {
        return None;
    }
    let s = std::f64::consts::SQRT_2 / mean_radius;
    Some(Mat3::new(
        s,
        0.0,
        -s * centroid.x,
        0.0,
        s,
        -s * centroid.y,
        0.0,
        0.0,
        1.0,
    ))
}

/// Apply a similarity whose bottom row is `(0, 0, 1)` — no division needed.
#[inline]
fn apply_similarity(t: &Mat3, p: Vec2) -> Vec2 {
    Vec2::new(t[(0, 0)] * p.x + t[(0, 2)], t[(1, 1)] * p.y + t[(1, 2)])
}

/// Direct linear transform over the supplied index subset.
fn dlt_indices(matches: &[(Vec2, Vec2)], indices: &[usize]) -> Option<Mat3> {
    if indices.len() < MIN_HOMOGRAPHY_MATCHES {
        return None;
    }
    let src: Vec<Vec2> = indices.iter().map(|&i| matches[i].0).collect();
    let dst: Vec<Vec2> = indices.iter().map(|&i| matches[i].1).collect();
    let t_src = normalizing_transform(&src)?;
    let t_dst = normalizing_transform(&dst)?;

    // nalgebra's SVD is thin: for an 8x9 design matrix `v_t` would be 8x9 and
    // would not contain the null direction at all. Pad to at least 9 rows.
    let rows = (2 * indices.len()).max(9);
    let mut a = DMatrix::<Scalar>::zeros(rows, 9);
    for (row, (&p, &q)) in src.iter().zip(dst.iter()).enumerate() {
        let p = apply_similarity(&t_src, p);
        let q = apply_similarity(&t_dst, q);
        let (r0, r1) = (2 * row, 2 * row + 1);
        a[(r0, 3)] = -p.x;
        a[(r0, 4)] = -p.y;
        a[(r0, 5)] = -1.0;
        a[(r0, 6)] = q.y * p.x;
        a[(r0, 7)] = q.y * p.y;
        a[(r0, 8)] = q.y;
        a[(r1, 0)] = p.x;
        a[(r1, 1)] = p.y;
        a[(r1, 2)] = 1.0;
        a[(r1, 6)] = -q.x * p.x;
        a[(r1, 7)] = -q.x * p.y;
        a[(r1, 8)] = -q.x;
    }

    let svd = a.svd(false, true);
    let v_t = svd.v_t.as_ref()?;
    let sv = &svd.singular_values;

    // Smallest singular value -> the null vector; second smallest -> the rank
    // check. nalgebra does not promise sorted singular values, so find both.
    let mut smallest = 0usize;
    for i in 1..sv.len() {
        if sv[i] < sv[smallest] {
            smallest = i;
        }
    }
    let mut second = usize::MAX;
    let mut largest = sv[0];
    for i in 0..sv.len() {
        if sv[i] > largest {
            largest = sv[i];
        }
        if i != smallest && (second == usize::MAX || sv[i] < sv[second]) {
            second = i;
        }
    }
    if second == usize::MAX || !(largest > 0.0) {
        return None;
    }
    if sv[second] < NULLSPACE_RATIO * largest {
        // Degenerate configuration (collinear or coincident): the solution is
        // not unique, so there is no honest answer to return.
        return None;
    }

    let h = v_t.row(smallest);
    let h_norm = Mat3::new(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]);
    let denorm = t_dst.try_inverse()? * h_norm * t_src;
    if !denorm.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some(normalize_homography(&denorm))
}

/// Normalised DLT over every supplied correspondence.
///
/// Returns the homography mapping the *first* element of each pair to the
/// second, scaled to unit Frobenius norm. `None` when there are fewer than
/// [`MIN_HOMOGRAPHY_MATCHES`] correspondences or the configuration is
/// degenerate (collinear or coincident points).
#[must_use]
pub fn estimate_homography(matches: &[(Vec2, Vec2)]) -> Option<Mat3> {
    let indices: Vec<usize> = (0..matches.len()).collect();
    dlt_indices(matches, &indices)
}

/// RANSAC over the normalised DLT, sampling minimal sets of four with
/// `DeterministicRng` (spec.md §6: *"Every RNG is seeded... RANSAC included"*).
///
/// `threshold` is the per-direction transfer error in pixels; a correspondence
/// is an inlier when its symmetric transfer error is within `2 * threshold^2`.
/// Returns the model refit on its inlier set, plus the inlier mask.
#[must_use]
pub fn estimate_homography_ransac(
    matches: &[(Vec2, Vec2)],
    threshold: Scalar,
    iterations: usize,
    rng: &mut DeterministicRng,
) -> Option<(Mat3, Vec<bool>)> {
    let n = matches.len();
    if n < MIN_HOMOGRAPHY_MATCHES {
        return None;
    }
    let cutoff = 2.0 * threshold * threshold;

    let mut sample = Vec::with_capacity(MINIMAL_SET);
    let mut best_inliers = 0usize;
    let mut best_cost = Scalar::INFINITY;
    let mut best_model: Option<Mat3> = None;
    let mut budget = iterations.max(1);

    let mut iter = 0usize;
    while iter < budget {
        iter += 1;
        rng.sample_distinct(n, MINIMAL_SET, &mut sample);
        let Some(h) = dlt_indices(matches, &sample) else {
            continue;
        };
        let Some(h_inv) = h.try_inverse() else {
            continue;
        };
        let mut count = 0usize;
        let mut cost = 0.0;
        for &(p, q) in matches {
            let e = symmetric_transfer_error(&h, &h_inv, p, q);
            if e <= cutoff {
                count += 1;
                cost += e;
            } else {
                cost += cutoff;
            }
        }
        if count > best_inliers || (count == best_inliers && cost < best_cost) {
            best_inliers = count;
            best_cost = cost;
            best_model = Some(h);
            // Adaptive stopping: once the inlier ratio is known, the number of
            // draws needed for RANSAC_CONFIDENCE is known too. Depends only on
            // the data, so the early exit stays reproducible for a given seed.
            let ratio = count as Scalar / n as Scalar;
            let p_good = ratio.powi(MINIMAL_SET as i32);
            if p_good > 0.0 && p_good < 1.0 {
                let needed = ((1.0 - RANSAC_CONFIDENCE).ln() / (1.0 - p_good).ln()).ceil();
                if needed.is_finite() && needed >= 0.0 {
                    budget = budget.min((needed as usize).max(iter));
                }
            } else if p_good >= 1.0 {
                budget = iter;
            }
        }
    }

    let model = best_model?;
    if best_inliers < MIN_HOMOGRAPHY_MATCHES {
        return None;
    }

    // Two refit passes: the minimal-set model gives a coarse inlier set, the
    // refit on it recovers points the coarse model just missed.
    let mut h = model;
    let mut mask = vec![false; n];
    for _ in 0..2 {
        let Some(h_inv) = h.try_inverse() else { break };
        let mut indices = Vec::with_capacity(n);
        for (i, &(p, q)) in matches.iter().enumerate() {
            let inlier = symmetric_transfer_error(&h, &h_inv, p, q) <= cutoff;
            mask[i] = inlier;
            if inlier {
                indices.push(i);
            }
        }
        match dlt_indices(matches, &indices) {
            Some(refit) => h = refit,
            None => break,
        }
    }

    // Final mask against the returned model, so caller-side inlier counts and
    // the model they belong to can never disagree.
    let h_inv = h.try_inverse()?;
    let mut count = 0usize;
    for (i, &(p, q)) in matches.iter().enumerate() {
        mask[i] = symmetric_transfer_error(&h, &h_inv, p, q) <= cutoff;
        count += usize::from(mask[i]);
    }
    if count < MIN_HOMOGRAPHY_MATCHES {
        return None;
    }
    Some((h, mask))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// A homography with a genuine projective part — an affine test matrix
    /// would not exercise the rows that carry `w`.
    fn known_homography() -> Mat3 {
        normalize_homography(&Mat3::new(
            1.08, -0.13, 24.0, 0.07, 0.94, -17.5, 2.5e-4, -1.1e-4, 1.0,
        ))
    }

    fn grid(n: usize) -> Vec<Vec2> {
        let mut pts = Vec::new();
        let side = (n as f64).sqrt().ceil() as usize;
        for i in 0..side {
            for j in 0..side {
                if pts.len() == n {
                    return pts;
                }
                // Deliberately irregular so the grid is not itself degenerate.
                let x = -300.0 + 640.0 * (i as f64 + 0.37 * (j as f64)) / side as f64;
                let y = -200.0 + 380.0 * (j as f64 + 0.21 * (i as f64)) / side as f64;
                pts.push(Vec2::new(x, y));
            }
        }
        pts
    }

    fn map_all(h: &Mat3, pts: &[Vec2]) -> Vec<(Vec2, Vec2)> {
        pts.iter()
            .map(|&p| (p, apply_homography(h, p).unwrap()))
            .collect()
    }

    #[test]
    fn hartley_normalisation_centres_and_scales() {
        let pts = grid(25);
        let t = normalizing_transform(&pts).unwrap();
        let mapped: Vec<Vec2> = pts.iter().map(|&p| apply_similarity(&t, p)).collect();
        let n = mapped.len() as f64;
        let centroid = mapped.iter().fold(Vec2::zeros(), |a, p| a + p) / n;
        let mean_radius = mapped.iter().map(|p| p.norm()).sum::<f64>() / n;
        assert_relative_eq!(centroid.x, 0.0, epsilon = 1e-12);
        assert_relative_eq!(centroid.y, 0.0, epsilon = 1e-12);
        assert_relative_eq!(mean_radius, std::f64::consts::SQRT_2, epsilon = 1e-12);
    }

    #[test]
    fn normalizing_transform_rejects_coincident_points() {
        let pts = vec![Vec2::new(3.0, 4.0); 8];
        assert!(normalizing_transform(&pts).is_none());
    }

    #[test]
    fn dlt_recovers_known_homography_from_four_correspondences() {
        let h_true = known_homography();
        let pts = vec![
            Vec2::new(-200.0, -150.0),
            Vec2::new(210.0, -140.0),
            Vec2::new(190.0, 160.0),
            Vec2::new(-180.0, 170.0),
        ];
        let h = estimate_homography(&map_all(&h_true, &pts)).unwrap();
        assert_relative_eq!(h, h_true, epsilon = 1e-10);
    }

    #[test]
    fn dlt_recovers_known_homography_from_many_correspondences() {
        let h_true = known_homography();
        let h = estimate_homography(&map_all(&h_true, &grid(64))).unwrap();
        assert_relative_eq!(h, h_true, epsilon = 1e-10);
    }

    #[test]
    fn dlt_rejects_fewer_than_four_correspondences() {
        let h_true = known_homography();
        let pts = grid(3);
        assert!(estimate_homography(&map_all(&h_true, &pts)).is_none());
    }

    #[test]
    fn dlt_rejects_collinear_correspondences() {
        // Degenerate by construction: four points on a line leave the
        // homography underdetermined, and returning "a" homography would be
        // worse than returning nothing.
        let h_true = known_homography();
        let pts: Vec<Vec2> = (0..6)
            .map(|i| Vec2::new(-250.0 + 100.0 * i as f64, 30.0))
            .collect();
        assert!(estimate_homography(&map_all(&h_true, &pts)).is_none());
    }

    #[test]
    fn homography_roundtrip_maps_points_back() {
        let h = known_homography();
        let h_inv = h.try_inverse().unwrap();
        for p in grid(16) {
            let q = apply_homography(&h, p).unwrap();
            let back = apply_homography(&h_inv, q).unwrap();
            assert_relative_eq!(back, p, epsilon = 1e-9);
        }
    }

    #[test]
    fn normalisation_is_invariant_to_scale_and_sign() {
        let h = known_homography();
        assert_relative_eq!(normalize_homography(&(h * -3.7)), h, epsilon = 1e-14);
        assert_relative_eq!(normalize_homography(&(h * 1e6)), h, epsilon = 1e-14);
    }

    #[test]
    fn ransac_recovers_homography_with_forty_percent_outliers() {
        let h_true = known_homography();
        let pts = grid(100);
        let mut rng = DeterministicRng::new("test-outliers", 20260801);
        let mut matches = map_all(&h_true, &pts);
        let mut truth = vec![true; matches.len()];
        for i in 0..matches.len() {
            if rng.uniform() < 0.40 {
                // A gross mismatch, not a nudge: this is what a wrong KLT
                // association looks like.
                matches[i].1 = Vec2::new(
                    rng.uniform_range(-320.0, 320.0),
                    rng.uniform_range(-240.0, 240.0),
                );
                truth[i] = false;
            }
        }
        let mut ransac_rng = DeterministicRng::new("test-ransac", 7);
        let (h, mask) = estimate_homography_ransac(&matches, 2.0, 2000, &mut ransac_rng).unwrap();
        assert_relative_eq!(h, h_true, epsilon = 1e-6);

        let agree = mask
            .iter()
            .zip(truth.iter())
            .filter(|(a, b)| a == b)
            .count();
        assert!(
            agree as f64 / mask.len() as f64 > 0.95,
            "inlier mask agreement {agree}/{}",
            mask.len()
        );
    }

    #[test]
    fn ransac_is_reproducible_for_a_seed() {
        let h_true = known_homography();
        let mut noise = DeterministicRng::new("test-noise", 3);
        let mut matches = map_all(&h_true, &grid(60));
        for m in matches.iter_mut() {
            m.1 += Vec2::new(noise.normal_with(0.0, 0.4), noise.normal_with(0.0, 0.4));
        }
        let run = |seed: u64| {
            let mut rng = DeterministicRng::new("t", seed);
            estimate_homography_ransac(&matches, 2.0, 300, &mut rng)
                .unwrap()
                .0
        };
        let a = run(11);
        let b = run(11);
        for i in 0..9 {
            assert_eq!(a[i].to_bits(), b[i].to_bits(), "seed 11 must be bit-exact");
        }
    }

    #[test]
    fn ransac_returns_none_when_no_consensus_exists() {
        let mut rng = DeterministicRng::new("t", 5);
        // Pure noise on both sides: any 4-point fit explains nothing else.
        let matches: Vec<(Vec2, Vec2)> = (0..40)
            .map(|_| {
                (
                    Vec2::new(
                        rng.uniform_range(-300.0, 300.0),
                        rng.uniform_range(-200.0, 200.0),
                    ),
                    Vec2::new(
                        rng.uniform_range(-300.0, 300.0),
                        rng.uniform_range(-200.0, 200.0),
                    ),
                )
            })
            .collect();
        let mut ransac_rng = DeterministicRng::new("t", 6);
        let out = estimate_homography_ransac(&matches, 0.5, 200, &mut ransac_rng);
        // Either nothing, or a model supported by only the minimal set.
        if let Some((_, mask)) = out {
            let inliers = mask.iter().filter(|m| **m).count();
            assert!(
                inliers <= 6,
                "random data must not produce consensus: {inliers}"
            );
        }
    }

    #[test]
    fn symmetric_transfer_error_is_zero_on_exact_correspondences() {
        let h = known_homography();
        let h_inv = h.try_inverse().unwrap();
        for p in grid(9) {
            let q = apply_homography(&h, p).unwrap();
            assert!(symmetric_transfer_error(&h, &h_inv, p, q) < 1e-18);
        }
    }
}
