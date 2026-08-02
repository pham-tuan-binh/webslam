//! Perspective-n-Point: camera pose from 3D-2D correspondences.
//!
//! The pose half of spec.md §4 L3 ("PnP against the active local map"). Three
//! layers, in the order RANSAC uses them:
//!
//! 1. [`solve_p3p`] — Grunert's three-point algorithm, up to four solutions.
//!    This is the minimal set, so it is what the sampler draws.
//! 2. [`solve_pnp_dlt`] — the linear n-point solve, used to seed refinement when
//!    the inlier set is large enough for it to be better conditioned than P3P.
//! 3. [`crate::motion_ba::refine_motion_only`] — the maximum-likelihood polish.
//!
//! [`pose_covariance`] is where the spec.md §6 L6 calibration claim starts. It
//! is `(J^T Σ^-1 J)^-1` evaluated at the solution in the `[translation;
//! rotation]` right-perturbation convention, not a scaled identity — an
//! estimator that reports a fudge factor is worse than one that reports nothing,
//! because the consumer cannot tell.
//!
//! All 2D inputs are **undistorted** pixel coordinates; see the module docs of
//! [`crate::motion_ba`].

use nalgebra::DMatrix;
use wslam_core::covariance::symmetrize;
use wslam_core::math::umeyama;
use wslam_core::{CameraIntrinsics, DeterministicRng, Mat3, Mat6, Scalar, Se3, So3, Vec2, Vec3};

use crate::motion_ba::{
    pinhole_only, pose_jacobian, project_pinhole, refine_motion_only, MotionBaConfig,
};

/// Minimal correspondence count for [`solve_pnp_ransac`]: three for the P3P
/// minimal set plus at least one to disambiguate between its solutions.
pub const MIN_RANSAC_POINTS: usize = 4;

/// Minimal correspondence count for [`solve_pnp_dlt`]. Eleven unknowns up to
/// scale, two equations per point.
pub const MIN_DLT_POINTS: usize = 6;

/// Outcome of a robust PnP solve.
#[derive(Debug, Clone)]
pub struct PnpResult {
    /// Estimated `T_world_camera`.
    pub pose: Se3,
    /// Per-correspondence inlier flags, in input order.
    pub inliers: Vec<bool>,
    /// Number of `true` entries in [`PnpResult::inliers`].
    pub inlier_count: usize,
    /// Mean reprojection error over the inliers, in pixels.
    pub mean_reprojection_error: Scalar,
}

/// Grunert's P3P (Haralick et al., IJCV 1994, "Review and Analysis of Solutions
/// of the Three Point Perspective Pose Estimation Problem").
///
/// Returns up to four `T_world_camera` solutions, each already satisfying
/// cheirality — the distances along the bearing rays are constrained positive,
/// so a solution with the scene behind the camera is never produced. Choosing
/// between the survivors needs a fourth point; see
/// [`solve_p3p_disambiguated`].
///
/// Returns an empty vector for the degenerate configurations: coincident world
/// points, collinear world points (the three constraint spheres then intersect
/// in a circle, so there is a one-parameter family of poses), and coincident
/// bearings.
#[must_use]
pub fn solve_p3p(points_3d: &[Vec3; 3], points_2d: &[Vec2; 3], k: &CameraIntrinsics) -> Vec<Se3> {
    let kp = pinhole_only(k);
    let bearings = [
        kp.unproject_bearing(points_2d[0]),
        kp.unproject_bearing(points_2d[1]),
        kp.unproject_bearing(points_2d[2]),
    ];

    let (p1, p2, p3) = (points_3d[0], points_3d[1], points_3d[2]);
    let side_a = (p3 - p2).norm(); // opposite vertex 1
    let side_b = (p3 - p1).norm(); // opposite vertex 2
    let side_c = (p2 - p1).norm(); // opposite vertex 3
    let scale = side_a.max(side_b).max(side_c);
    if scale <= 0.0 || side_a < 1e-9 * scale || side_b < 1e-9 * scale || side_c < 1e-9 * scale {
        return Vec::new();
    }
    // Collinear world points leave the pose underdetermined. Twice the triangle
    // area over the product of two sides is sin of the enclosed angle, so this
    // is a scale-free test.
    if (p2 - p1).cross(&(p3 - p1)).norm() < 1e-8 * side_b * side_c {
        return Vec::new();
    }

    let cos_alpha = bearings[1].dot(&bearings[2]).clamp(-1.0, 1.0);
    let cos_beta = bearings[0].dot(&bearings[2]).clamp(-1.0, 1.0);
    let cos_gamma = bearings[0].dot(&bearings[1]).clamp(-1.0, 1.0);
    if (1.0 - cos_alpha.abs()) < 1e-14
        || (1.0 - cos_beta.abs()) < 1e-14
        || (1.0 - cos_gamma.abs()) < 1e-14
    {
        return Vec::new(); // two rays coincide: no triangle in the image
    }

    // Grunert's substitution s2 = u s1, s3 = v s1 turns the three cosine-rule
    // equations into a quartic in v. See the derivation in the tests: u is a
    // rational function of v, and eliminating it clears a degree-2 denominator.
    let m = (side_a * side_a - side_c * side_c) / (side_b * side_b);
    let n = (side_c * side_c) / (side_b * side_b);

    let (n2, n1, n0) = (m - 1.0, -2.0 * m * cos_beta, m + 1.0);
    let (d1, d0) = (-2.0 * cos_alpha, 2.0 * cos_gamma);
    let (g2, g1, g0) = (-n, 2.0 * n * cos_beta, 1.0 - n);
    let kk = -2.0 * cos_gamma;

    let quartic = [
        n0 * n0 + kk * n0 * d0 + g0 * d0 * d0,
        2.0 * n1 * n0 + kk * (n1 * d0 + n0 * d1) + g1 * d0 * d0 + 2.0 * g0 * d1 * d0,
        n1 * n1
            + 2.0 * n2 * n0
            + kk * (n2 * d0 + n1 * d1)
            + g2 * d0 * d0
            + 2.0 * g1 * d1 * d0
            + g0 * d1 * d1,
        2.0 * n2 * n1 + kk * n2 * d1 + 2.0 * g2 * d1 * d0 + g1 * d1 * d1,
        n2 * n2 + g2 * d1 * d1,
    ];

    let mut out = Vec::with_capacity(4);
    for v in solve_quartic(&quartic) {
        if !(v.is_finite() && v > 0.0) {
            continue;
        }
        let denom = 1.0 + v * v - 2.0 * v * cos_beta;
        if denom <= 1e-12 {
            continue;
        }
        let s1 = (side_b * side_b / denom).sqrt();

        // u from the eliminated relation, with a fallback for the pole at
        // cos(gamma) = v cos(alpha) where that expression is 0/0.
        let d_of_v = d1 * v + d0;
        let candidates_u = if d_of_v.abs() > 1e-9 {
            let n_of_v = n2 * v * v + n1 * v + n0;
            vec![n_of_v / d_of_v]
        } else {
            // u^2 - 2 u cos(gamma) + (1 - n(1 + v^2 - 2 v cos(beta))) = 0
            let c = 1.0 - n * denom;
            let disc = cos_gamma * cos_gamma - c;
            if disc < 0.0 {
                continue;
            }
            let root = disc.sqrt();
            vec![cos_gamma + root, cos_gamma - root]
        };

        for u in candidates_u {
            if !(u.is_finite() && u > 0.0) {
                continue;
            }
            let cam = [
                bearings[0] * s1,
                bearings[1] * (u * s1),
                bearings[2] * (v * s1),
            ];
            // Reject branches that do not reproduce the triangle: the fallback
            // above and any surviving numerical noise both show up here.
            let fit_a = ((cam[2] - cam[1]).norm() - side_a).abs();
            let fit_c = ((cam[1] - cam[0]).norm() - side_c).abs();
            if fit_a > 1e-6 * scale || fit_c > 1e-6 * scale {
                continue;
            }
            // Horn absolute orientation, scale locked: three non-collinear
            // points with matching inter-point distances fix the transform.
            let Some(align) = umeyama(points_3d, &cam, false) else {
                continue;
            };
            if align.rmse > 1e-7 * scale {
                continue;
            }
            let t_cam_world = Se3::new(align.transform.rotation(), align.transform.translation());
            out.push(t_cam_world.inverse());
        }
    }
    out
}

/// P3P over the first three correspondences, disambiguated by the reprojection
/// error of every remaining correspondence.
///
/// Needs at least four points: the fourth is what distinguishes the P3P
/// solutions, all of which fit the first three exactly by construction.
#[must_use]
pub fn solve_p3p_disambiguated(
    points_3d: &[Vec3],
    points_2d: &[Vec2],
    k: &CameraIntrinsics,
) -> Option<Se3> {
    if points_3d.len() < 4 || points_2d.len() < points_3d.len() {
        return None;
    }
    let triple_3d = [points_3d[0], points_3d[1], points_3d[2]];
    let triple_2d = [points_2d[0], points_2d[1], points_2d[2]];
    let solutions = solve_p3p(&triple_3d, &triple_2d, k);
    select_by_reprojection(&solutions, &points_3d[3..], &points_2d[3..], k)
}

/// Pick the pose with the lowest total reprojection error over a validation set,
/// rejecting any pose that puts a validation point behind the camera.
#[must_use]
pub fn select_by_reprojection(
    candidates: &[Se3],
    points_3d: &[Vec3],
    points_2d: &[Vec2],
    k: &CameraIntrinsics,
) -> Option<Se3> {
    let kp = pinhole_only(k);
    let mut best: Option<(Scalar, Se3)> = None;
    for pose in candidates {
        let inv = pose.inverse();
        let mut total = 0.0;
        let mut ok = true;
        for (p, px) in points_3d.iter().zip(points_2d.iter()) {
            match project_pinhole(&kp, &inv.act(p)) {
                Some(pred) => total += (pred - px).norm(),
                None => {
                    ok = false; // cheirality: this candidate is not physical
                    break;
                }
            }
        }
        if ok && best.as_ref().is_none_or(|(b, _)| total < *b) {
            best = Some((total, *pose));
        }
    }
    best.map(|(_, p)| p)
}

/// Direct linear transform PnP.
///
/// Solves for the 3x4 camera matrix in the null space of the stacked
/// correspondence constraints, then projects its left 3x3 block to the nearest
/// rotation. Isotropic Hartley normalisation is applied to the 3D points, which
/// matters more here than it does for a homography: world coordinates arrive in
/// metres with an arbitrary origin, so an unnormalised design matrix routinely
/// has a condition number in the millions.
///
/// This is a *seed*, not an answer — the DLT ignores the rotation constraint
/// and is biased under noise. Follow it with
/// [`crate::motion_ba::refine_motion_only`].
#[must_use]
pub fn solve_pnp_dlt(points_3d: &[Vec3], points_2d: &[Vec2], k: &CameraIntrinsics) -> Option<Se3> {
    let n = points_3d.len();
    if n < MIN_DLT_POINTS || points_2d.len() < n {
        return None;
    }
    let kp = pinhole_only(k);

    let centroid: Vec3 = points_3d.iter().sum::<Vec3>() / n as Scalar;
    let mean_dist = points_3d
        .iter()
        .map(|p| (p - centroid).norm())
        .sum::<Scalar>()
        / n as Scalar;
    if !(mean_dist.is_finite() && mean_dist > 1e-12) {
        return None; // all landmarks coincident
    }
    let s = (3.0 as Scalar).sqrt() / mean_dist;

    let mut a = DMatrix::<Scalar>::zeros(2 * n, 12);
    for i in 0..n {
        let q = (points_3d[i] - centroid) * s;
        let z = kp.unproject_normalized(points_2d[i]);
        let xh = [q.x, q.y, q.z, 1.0];
        for (c, &v) in xh.iter().enumerate() {
            a[(2 * i, c)] = -v;
            a[(2 * i, 8 + c)] = z.x * v;
            a[(2 * i + 1, 4 + c)] = -v;
            a[(2 * i + 1, 8 + c)] = z.y * v;
        }
    }

    let v = smallest_right_singular_vector(&a)?;
    // Row-major 3x4, still expressed for normalised world coordinates.
    let mut p_norm = nalgebra::Matrix3x4::<Scalar>::zeros();
    for r in 0..3 {
        for c in 0..4 {
            p_norm[(r, c)] = v[r * 4 + c];
        }
    }
    // Undo the 3D normalisation: X_norm = s (X - centroid).
    let mut u = nalgebra::Matrix4::<Scalar>::identity();
    u[(0, 0)] = s;
    u[(1, 1)] = s;
    u[(2, 2)] = s;
    u[(0, 3)] = -s * centroid.x;
    u[(1, 3)] = -s * centroid.y;
    u[(2, 3)] = -s * centroid.z;
    let mut p = p_norm * u;

    // Overall sign: the scene must be in front of the camera.
    let depth =
        p[(2, 0)] * centroid.x + p[(2, 1)] * centroid.y + p[(2, 2)] * centroid.z + p[(2, 3)];
    if depth < 0.0 {
        p = -p;
    }

    let m: Mat3 = p.fixed_view::<3, 3>(0, 0).into_owned();
    let svd = m.svd(true, true);
    let (u_m, vt_m) = (svd.u?, svd.v_t?);
    let rot = u_m * vt_m;
    if rot.determinant() < 0.0 {
        return None; // a reflection means the linear solve did not find a pose
    }
    let sv = svd.singular_values;
    let lambda = (sv[0] + sv[1] + sv[2]) / 3.0;
    if !(lambda.is_finite() && lambda > 1e-12) {
        return None;
    }
    let t_cw = Vec3::new(p[(0, 3)], p[(1, 3)], p[(2, 3)]) / lambda;
    let so3 = So3::from_quaternion(nalgebra::UnitQuaternion::from_rotation_matrix(
        &nalgebra::Rotation3::from_matrix_unchecked(rot),
    ));
    Some(Se3::new(so3, t_cw).inverse())
}

/// Robust PnP: P3P inside RANSAC over a seeded generator, then a
/// maximum-likelihood polish over the inliers.
///
/// spec.md §6 makes this non-negotiable: *"Every RNG is seeded and the seed is
/// logged. RANSAC included."* The minimal sets come from `rng`, so the same seed
/// and the same input produce a bit-identical result — which is what makes the
/// replay harness a regression wall rather than a mood ring.
///
/// The iteration count adapts to the inlier ratio found so far,
/// `N = log(1-p) / log(1-w^3)` at `p = 0.99`, capped by `iterations`. Cheap
/// scenes exit in a handful of samples; the cap bounds the pathological case.
#[must_use]
pub fn solve_pnp_ransac(
    points_3d: &[Vec3],
    points_2d: &[Vec2],
    k: &CameraIntrinsics,
    threshold_px: Scalar,
    iterations: usize,
    rng: &mut DeterministicRng,
) -> Option<PnpResult> {
    let n = points_3d.len();
    if n < MIN_RANSAC_POINTS || points_2d.len() < n || iterations == 0 || threshold_px <= 0.0 {
        return None;
    }
    let kp = pinhole_only(k);

    let mut best_pose: Option<Se3> = None;
    let mut best_count = 0usize;
    let mut best_error = Scalar::INFINITY;
    let mut needed = iterations;
    let mut sample = Vec::with_capacity(3);
    let mut scratch = vec![false; n];

    for iteration in 0..iterations {
        if iteration >= needed {
            break;
        }
        rng.sample_distinct(n, 3, &mut sample);
        if sample.len() < 3 {
            break;
        }
        let triple_3d = [
            points_3d[sample[0]],
            points_3d[sample[1]],
            points_3d[sample[2]],
        ];
        let triple_2d = [
            points_2d[sample[0]],
            points_2d[sample[1]],
            points_2d[sample[2]],
        ];
        for pose in solve_p3p(&triple_3d, &triple_2d, &kp) {
            let (count, error) =
                classify(&pose, points_3d, points_2d, &kp, threshold_px, &mut scratch);
            // Tie-break on total inlier error so the winner is reproducible even
            // when two minimal sets agree on the inlier count.
            if count > best_count || (count == best_count && error < best_error) {
                best_count = count;
                best_error = error;
                best_pose = Some(pose);
            }
        }
        if best_count > 0 {
            let w = best_count as Scalar / n as Scalar;
            needed = adaptive_iterations(w, 3, 0.99).min(iterations);
        }
    }

    let mut pose = best_pose?;
    if best_count < MIN_RANSAC_POINTS {
        return None;
    }

    // Three rounds of "refit on the consensus set, then re-classify". The
    // refinement is masked to the current inliers rather than robustified over
    // everything: Huber bounds an outlier's influence but does not remove it,
    // and at a 50% contamination rate the surviving pull is enough to move the
    // pose by centimetres.
    let mut inliers = vec![false; n];
    classify(&pose, points_3d, points_2d, &kp, threshold_px, &mut inliers);
    let ba_config = MotionBaConfig {
        huber_delta_px: threshold_px,
        outlier_threshold_px: threshold_px,
        max_iterations: 15,
        ..MotionBaConfig::default()
    };
    for _ in 0..3 {
        let seeded = better_seed(&pose, points_3d, points_2d, &inliers, &kp);
        let Some(refined) = refine_motion_only(
            &seeded,
            points_3d,
            points_2d,
            &kp,
            Some(&inliers),
            &ba_config,
        ) else {
            break;
        };
        if refined.inlier_count < MIN_RANSAC_POINTS {
            break;
        }
        pose = refined.pose;
        let (count, _) = classify(&pose, points_3d, points_2d, &kp, threshold_px, &mut inliers);
        if count < MIN_RANSAC_POINTS {
            return None;
        }
    }

    let (inlier_count, error_sum) =
        classify(&pose, points_3d, points_2d, &kp, threshold_px, &mut inliers);
    if inlier_count < MIN_RANSAC_POINTS {
        return None;
    }
    Some(PnpResult {
        pose,
        inliers,
        inlier_count,
        mean_reprojection_error: error_sum / inlier_count as Scalar,
    })
}

/// 6x6 pose covariance from the Gauss-Newton normal equations at `pose`.
///
/// `(J^T Σ^-1 J)^-1` with `Σ = pixel_sigma^2 I` per measurement, which reduces
/// to `pixel_sigma^2 (J^T J)^-1`. Ordered `[translation; rotation]` in the
/// right-perturbation tangent space at `pose`, matching
/// [`wslam_core::covariance::pose_nees`] and the 6x6 the public API promises.
///
/// `points_2d` is deliberately absent: the Fisher information of a
/// least-squares problem with additive Gaussian noise depends only on the
/// geometry and the linearisation point, not on the values that were measured.
/// Passing the measurements would suggest otherwise.
///
/// Returns `None` when fewer than three landmarks are visible or the
/// information matrix is singular — a degenerate geometry must not be papered
/// over with a large-but-finite number, because a consumer cannot distinguish
/// that from a real answer.
#[must_use]
pub fn pose_covariance(
    pose: &Se3,
    points_3d: &[Vec3],
    k: &CameraIntrinsics,
    pixel_sigma: Scalar,
    inliers: Option<&[bool]>,
) -> Option<Mat6> {
    if !(pixel_sigma.is_finite() && pixel_sigma > 0.0) {
        return None;
    }
    let kp = pinhole_only(k);
    let mut information = Mat6::zeros();
    let mut used = 0usize;
    for (i, p) in points_3d.iter().enumerate() {
        if let Some(m) = inliers {
            if !m.get(i).copied().unwrap_or(false) {
                continue;
            }
        }
        if let Some((_, j)) = pose_jacobian(pose, p, &kp) {
            information += j.transpose() * j;
            used += 1;
        }
    }
    if used < 3 {
        return None;
    }
    let cov = symmetrize(&(information.try_inverse()? * (pixel_sigma * pixel_sigma)));
    let sane = cov.iter().all(|v| v.is_finite()) && (0..6).all(|i| cov[(i, i)] > 0.0);
    sane.then_some(cov)
}

/// RANSAC iteration count for a given inlier ratio, sample size and confidence.
///
/// `N = log(1 - confidence) / log(1 - w^s)`, clamped to at least one.
///
/// The two limits are not symmetric and conflating them is a silent
/// correctness bug rather than a rounding one:
///
/// * `w -> 1` makes `w^s -> 1` and `log(1 - w^s) -> -inf`, so `N -> 1`. One
///   sample suffices — every draw is clean.
/// * `w -> 0` makes `w^s -> 0` and `log(1 - w^s) -> 0^-`, so `N -> +inf`. No
///   finite budget is enough, and the honest answer is "keep going".
///
/// A single `|denominator| < eps -> return 1` guard collapses *both* limits to
/// one iteration, which turns a caller's adaptive cap into "give up after the
/// first hypothesis" precisely when the first hypothesis was garbage. The
/// denominator is therefore evaluated with `ln_1p`, which stays accurate as
/// `w^s -> 0`, and only the `-inf` branch shortcuts to one.
#[must_use]
pub fn adaptive_iterations(inlier_ratio: Scalar, sample_size: u32, confidence: Scalar) -> usize {
    let w = inlier_ratio.clamp(1e-6, 1.0 - 1e-9);
    let p_clean = w.powi(sample_size as i32);
    // ln(1 - p) evaluated as ln_1p(-p): exact to full precision for tiny p,
    // where `(1.0 - p).ln()` cancels to zero.
    let denom = (-p_clean).ln_1p();
    if !denom.is_finite() {
        return 1; // p_clean == 1: the data is noiseless.
    }
    if denom == 0.0 {
        return usize::MAX; // p_clean underflowed: unbounded.
    }
    let n = (1.0 - confidence).ln() / denom;
    if n.is_nan() || n <= 1.0 {
        1
    } else {
        // Saturating float->int cast: an astronomically large N becomes
        // `usize::MAX`, i.e. "the caller's cap is the only limit".
        n.ceil() as usize
    }
}

/// Fill `out` with the inlier flags for `pose` and return
/// `(count, sum_of_inlier_errors)`.
fn classify(
    pose: &Se3,
    points_3d: &[Vec3],
    points_2d: &[Vec2],
    k: &CameraIntrinsics,
    threshold_px: Scalar,
    out: &mut [bool],
) -> (usize, Scalar) {
    let inv = pose.inverse();
    let mut count = 0usize;
    let mut sum = 0.0;
    for (i, flag) in out.iter_mut().enumerate() {
        *flag = false;
        let Some(pred) = project_pinhole(k, &inv.act(&points_3d[i])) else {
            continue;
        };
        let e = (pred - points_2d[i]).norm();
        if e <= threshold_px {
            *flag = true;
            count += 1;
            sum += e;
        }
    }
    (count, sum)
}

/// Whichever of the current pose and a fresh DLT over the consensus set has the
/// lower reprojection cost.
///
/// The DLT is usually the better starting point once the inlier set is large,
/// but it ignores the rotation constraint, so on a marginal set it can be worse
/// than the minimal-sample pose it would replace. Measuring is cheaper than
/// guessing.
fn better_seed(
    current: &Se3,
    points_3d: &[Vec3],
    points_2d: &[Vec2],
    inliers: &[bool],
    k: &CameraIntrinsics,
) -> Se3 {
    let cost = |p: &Se3| {
        crate::motion_ba::normal_equations(p, points_3d, points_2d, k, Some(inliers), None).cost
    };
    match seed_from_inliers(points_3d, points_2d, inliers, k) {
        Some(dlt) if cost(&dlt) < cost(current) => dlt,
        _ => *current,
    }
}

/// Re-seed the refinement from a DLT over the current inlier set, which is
/// better conditioned than a three-point solution once the set is large.
fn seed_from_inliers(
    points_3d: &[Vec3],
    points_2d: &[Vec2],
    inliers: &[bool],
    k: &CameraIntrinsics,
) -> Option<Se3> {
    let sub_3d: Vec<Vec3> = points_3d
        .iter()
        .zip(inliers)
        .filter_map(|(p, &ok)| ok.then_some(*p))
        .collect();
    if sub_3d.len() < MIN_DLT_POINTS {
        return None;
    }
    let sub_2d: Vec<Vec2> = points_2d
        .iter()
        .zip(inliers)
        .filter_map(|(p, &ok)| ok.then_some(*p))
        .collect();
    solve_pnp_dlt(&sub_3d, &sub_2d, k)
}

/// Right singular vector of the smallest singular value.
///
/// `nalgebra` computes the thin SVD, so `V^T` only has `min(rows, cols)` rows;
/// zero-padding to a tall matrix is what makes the null vector reachable when
/// there are exactly as many constraints as unknowns minus one.
fn smallest_right_singular_vector(a: &DMatrix<Scalar>) -> Option<nalgebra::DVector<Scalar>> {
    let cols = a.ncols();
    let padded = if a.nrows() >= cols {
        a.clone()
    } else {
        let mut m = DMatrix::<Scalar>::zeros(cols, cols);
        m.view_mut((0, 0), (a.nrows(), cols)).copy_from(a);
        m
    };
    let svd = padded.svd(false, true);
    let v_t = svd.v_t?;
    let idx = svd
        .singular_values
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)?;
    Some(v_t.row(idx).transpose())
}

/// Real roots of `c[4] x^4 + c[3] x^3 + c[2] x^2 + c[1] x + c[0]`, via Ferrari's
/// resolvent cubic and a Newton polish.
///
/// The polish is not decoration: the resolvent introduces cancellation, and P3P
/// feeds these roots straight into a square root, so a root that is right to
/// only six digits produces a pose that is visibly wrong.
fn solve_quartic(c: &[Scalar; 5]) -> Vec<Scalar> {
    let scale = c.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    if !scale.is_finite() || scale == 0.0 {
        return Vec::new();
    }
    if c[4].abs() < 1e-14 * scale {
        return solve_cubic(c[3], c[2], c[1], c[0]);
    }
    let (a, b, cc, d) = (c[3] / c[4], c[2] / c[4], c[1] / c[4], c[0] / c[4]);

    // Depressed quartic y^4 + p y^2 + q y + r with x = y - a/4.
    let shift = a / 4.0;
    let p = b - 3.0 / 8.0 * a * a;
    let q = cc - 0.5 * a * b + a * a * a / 8.0;
    let r = d - 0.25 * a * cc + a * a * b / 16.0 - 3.0 / 256.0 * a * a * a * a;

    let mut roots = Vec::with_capacity(4);
    if q.abs() < 1e-14 * (1.0 + p.abs() + r.abs()) {
        // Biquadratic.
        let disc = p * p - 4.0 * r;
        if disc >= 0.0 {
            let s = disc.sqrt();
            for z in [(-p + s) * 0.5, (-p - s) * 0.5] {
                if z >= 0.0 {
                    let y = z.sqrt();
                    roots.push(y - shift);
                    roots.push(-y - shift);
                }
            }
        }
    } else {
        // (y^2 + p/2 + m)^2 = 2m (y - q/(4m))^2 is a perfect square exactly when
        // 8 m^3 + 8 p m^2 + (2p^2 - 8r) m - q^2 = 0. That cubic is negative at
        // m = 0 and grows without bound, so a positive real root always exists.
        let cubic = solve_cubic_monic(p, (p * p - 4.0 * r) / 4.0, -q * q / 8.0);
        let m = cubic
            .into_iter()
            .filter(|m| *m > 1e-14)
            .fold(0.0_f64, f64::max);
        if m > 0.0 {
            let s = (2.0 * m).sqrt();
            let base = p * 0.5 + m;
            for (lin, konst) in [(-s, base + q / (2.0 * s)), (s, base - q / (2.0 * s))] {
                let disc = lin * lin - 4.0 * konst;
                if disc >= 0.0 {
                    let sq = disc.sqrt();
                    roots.push((-lin + sq) * 0.5 - shift);
                    roots.push((-lin - sq) * 0.5 - shift);
                }
            }
        }
    }

    for root in roots.iter_mut() {
        for _ in 0..8 {
            let f = ((((c[4] * *root) + c[3]) * *root + c[2]) * *root + c[1]) * *root + c[0];
            let df = (((4.0 * c[4] * *root) + 3.0 * c[3]) * *root + 2.0 * c[2]) * *root + c[1];
            if df.abs() < 1e-300 {
                break;
            }
            let step = f / df;
            *root -= step;
            if step.abs() < 1e-16 * (1.0 + root.abs()) {
                break;
            }
        }
    }
    roots.retain(|r| r.is_finite());
    // A repeated root reaches here twice; P3P treats each survivor as a
    // distinct pose, and "up to four solutions" should mean four *distinct* ones.
    roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    roots.dedup_by(|a, b| (*a - *b).abs() <= 1e-9 * (1.0 + a.abs()));
    roots
}

/// Real roots of `a3 x^3 + a2 x^2 + a1 x + a0`, degrading to the quadratic or
/// linear case when the leading coefficients vanish.
fn solve_cubic(a3: Scalar, a2: Scalar, a1: Scalar, a0: Scalar) -> Vec<Scalar> {
    if a3.abs() < 1e-300 {
        return solve_quadratic(a2, a1, a0);
    }
    solve_cubic_monic(a2 / a3, a1 / a3, a0 / a3)
}

/// Real roots of the monic cubic `x^3 + b x^2 + c x + d`.
fn solve_cubic_monic(b: Scalar, c: Scalar, d: Scalar) -> Vec<Scalar> {
    // Depressed cubic t^3 + p t + q, x = t - b/3.
    let shift = b / 3.0;
    let p = c - b * b / 3.0;
    let q = 2.0 * b * b * b / 27.0 - b * c / 3.0 + d;
    let half_q = q * 0.5;
    let third_p = p / 3.0;
    let disc = half_q * half_q + third_p * third_p * third_p;

    if disc > 0.0 {
        let s = disc.sqrt();
        let t = (-half_q + s).cbrt() + (-half_q - s).cbrt();
        vec![t - shift]
    } else if disc.abs() <= 1e-300 {
        let u = (-half_q).cbrt();
        vec![2.0 * u - shift, -u - shift]
    } else {
        let m = (-third_p).sqrt();
        let phi = (-half_q / (m * m * m)).clamp(-1.0, 1.0).acos() / 3.0;
        let tau = std::f64::consts::TAU / 3.0;
        vec![
            2.0 * m * phi.cos() - shift,
            2.0 * m * (phi - tau).cos() - shift,
            2.0 * m * (phi + tau).cos() - shift,
        ]
    }
}

fn solve_quadratic(a: Scalar, b: Scalar, c: Scalar) -> Vec<Scalar> {
    if a.abs() < 1e-300 {
        return if b.abs() < 1e-300 {
            Vec::new()
        } else {
            vec![-c / b]
        };
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        Vec::new()
    } else {
        let s = disc.sqrt();
        vec![(-b + s) / (2.0 * a), (-b - s) / (2.0 * a)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use wslam_core::covariance::ConsistencyAccumulator;
    use wslam_core::stats::nees_bounds;
    use wslam_core::Vec6;

    fn intrinsics() -> CameraIntrinsics {
        CameraIntrinsics::from_focal(520.0, 640, 480)
    }

    fn truth_pose() -> Se3 {
        Se3::new(
            So3::exp(&Vec3::new(0.06, -0.11, 0.04)),
            Vec3::new(0.35, -0.2, -2.5),
        )
    }

    fn scene(pose: &Se3, k: &CameraIntrinsics, n: usize, seed: u64) -> (Vec<Vec3>, Vec<Vec2>) {
        let kp = pinhole_only(k);
        let mut rng = DeterministicRng::new("scene", seed);
        let (mut p3, mut p2) = (Vec::new(), Vec::new());
        while p3.len() < n {
            let p = Vec3::new(
                rng.uniform_range(-1.8, 1.8),
                rng.uniform_range(-1.4, 1.4),
                rng.uniform_range(2.0, 7.0),
            );
            let cam = pose.inverse().act(&p);
            if cam.z < 0.5 {
                continue;
            }
            match project_pinhole(&kp, &cam) {
                Some(px) if kp.contains(px, 8.0) => {
                    p3.push(p);
                    p2.push(px);
                }
                _ => {}
            }
        }
        (p3, p2)
    }

    #[test]
    fn quartic_solver_recovers_planted_roots() {
        // (x-1)(x-2)(x+3)(x-0.25) expanded.
        let roots = [1.0, 2.0, -3.0, 0.25];
        let mut c = [0.0; 5];
        c[4] = 1.0;
        // Build coefficients by convolution so the test does not restate the
        // solver's own algebra.
        let mut poly = vec![1.0];
        for r in roots {
            let mut next = vec![0.0; poly.len() + 1];
            for (i, v) in poly.iter().enumerate() {
                next[i] += *v;
                next[i + 1] -= v * r;
            }
            poly = next;
        }
        for (i, v) in poly.iter().enumerate() {
            c[4 - i] = *v;
        }
        let mut found = solve_quartic(&c);
        found.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut want = roots.to_vec();
        want.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(found.len(), 4, "{found:?}");
        for (f, w) in found.iter().zip(want.iter()) {
            assert_relative_eq!(f, w, epsilon = 1e-10);
        }
    }

    #[test]
    fn quartic_solver_handles_two_real_roots() {
        // (x^2 + 1)(x - 4)(x + 1) : only two real roots.
        let c = [-4.0, -3.0, -3.0, -3.0, 1.0];
        let mut found = solve_quartic(&c);
        found.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(found.len(), 2, "{found:?}");
        assert_relative_eq!(found[0], -1.0, epsilon = 1e-10);
        assert_relative_eq!(found[1], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn p3p_solutions_contain_the_truth_exactly() {
        let k = intrinsics();
        let truth = truth_pose();
        let (p3, p2) = scene(&truth, &k, 3, 11);
        let solutions = solve_p3p(&[p3[0], p3[1], p3[2]], &[p2[0], p2[1], p2[2]], &k);
        assert!(
            !solutions.is_empty() && solutions.len() <= 4,
            "{} solutions",
            solutions.len()
        );
        let best = solutions
            .iter()
            .map(|s| s.minus(&truth).norm())
            .fold(Scalar::INFINITY, f64::min);
        assert!(best < 1e-9, "closest P3P solution is {best} from truth");
        // Every returned solution must reproduce the three input pixels.
        for s in &solutions {
            for i in 0..3 {
                let e = crate::motion_ba::reprojection_error(s, &p3[i], p2[i], &k).unwrap();
                assert!(e < 1e-7, "solution does not fit its own minimal set: {e}");
            }
        }
    }

    #[test]
    fn p3p_rejects_collinear_world_points() {
        let k = intrinsics();
        let truth = truth_pose();
        let base = Vec3::new(-0.5, 0.2, 4.0);
        let dir = Vec3::new(1.0, 0.3, 0.4).normalize();
        let p3 = [base, base + dir * 0.8, base + dir * 1.9];
        let kp = pinhole_only(&k);
        let inv = truth.inverse();
        let p2 = [
            project_pinhole(&kp, &inv.act(&p3[0])).unwrap(),
            project_pinhole(&kp, &inv.act(&p3[1])).unwrap(),
            project_pinhole(&kp, &inv.act(&p3[2])).unwrap(),
        ];
        assert!(solve_p3p(&p3, &p2, &k).is_empty());
    }

    #[test]
    fn p3p_rejects_coincident_world_points() {
        let k = intrinsics();
        let p = Vec3::new(0.1, 0.2, 3.0);
        let px = Vec2::new(320.0, 240.0);
        assert!(solve_p3p(&[p, p, p], &[px, px, px], &k).is_empty());
    }

    #[test]
    fn p3p_disambiguated_picks_the_true_pose() {
        let k = intrinsics();
        let truth = truth_pose();
        let (p3, p2) = scene(&truth, &k, 6, 12);
        let pose = solve_p3p_disambiguated(&p3, &p2, &k).unwrap();
        assert!(pose.minus(&truth).norm() < 1e-9);
    }

    #[test]
    fn dlt_recovers_a_known_pose_from_noise_free_points() {
        let k = intrinsics();
        let truth = truth_pose();
        let (p3, p2) = scene(&truth, &k, 20, 13);
        let pose = solve_pnp_dlt(&p3, &p2, &k).unwrap();
        assert!(pose.minus(&truth).norm() < 1e-8, "{:?}", pose.minus(&truth));
    }

    #[test]
    fn dlt_needs_six_points() {
        let k = intrinsics();
        let truth = truth_pose();
        let (p3, p2) = scene(&truth, &k, 5, 14);
        assert!(solve_pnp_dlt(&p3, &p2, &k).is_none());
    }

    /// Half the correspondences are garbage. RANSAC must find the pose and name
    /// the outliers.
    #[test]
    fn ransac_survives_fifty_percent_gross_outliers() {
        let k = intrinsics();
        let truth = truth_pose();
        let (p3, mut p2) = scene(&truth, &k, 80, 15);
        let mut rng = DeterministicRng::new("corrupt", 4242);
        let mut is_outlier = [false; 80];
        for (i, px) in p2.iter_mut().enumerate() {
            if i % 2 == 0 {
                *px = Vec2::new(rng.uniform_range(0.0, 640.0), rng.uniform_range(0.0, 480.0));
                is_outlier[i] = true;
            }
        }
        let mut rng = DeterministicRng::new("ransac", 7);
        let r = solve_pnp_ransac(&p3, &p2, &k, 2.0, 2000, &mut rng).unwrap();
        assert!(
            r.pose.minus(&truth).norm() < 1e-6,
            "pose error {:?}",
            r.pose.minus(&truth)
        );
        // The inlier set must be exactly the clean half. A uniformly-placed
        // random pixel lands within 2 px of its true projection with
        // probability ~4e-5, so a stray acceptance is a real failure.
        for (i, &out) in is_outlier.iter().enumerate() {
            assert_eq!(!out, r.inliers[i], "correspondence {i} misclassified");
        }
        assert_eq!(r.inlier_count, 40);
        assert!(r.mean_reprojection_error < 1e-6);
    }

    #[test]
    fn ransac_is_bit_identical_across_runs_with_the_same_seed() {
        let k = intrinsics();
        let truth = truth_pose();
        let (p3, mut p2) = scene(&truth, &k, 60, 16);
        let mut noise = DeterministicRng::new("noise", 3);
        for (i, px) in p2.iter_mut().enumerate() {
            if i % 3 == 0 {
                *px = Vec2::new(
                    noise.uniform_range(0.0, 640.0),
                    noise.uniform_range(0.0, 480.0),
                );
            } else {
                *px += Vec2::new(noise.normal() * 0.4, noise.normal() * 0.4);
            }
        }
        let run = |seed: u64| {
            let mut rng = DeterministicRng::new("ransac", seed);
            solve_pnp_ransac(&p3, &p2, &k, 2.0, 500, &mut rng).unwrap()
        };
        let a = run(20260801);
        let b = run(20260801);
        // Bit-identical, not merely close: spec.md §6 requires the replay to be
        // reproducible bit-for-bit.
        assert_eq!(a.pose.log(), b.pose.log());
        assert_eq!(a.inliers, b.inliers);
        assert_eq!(
            a.mean_reprojection_error.to_bits(),
            b.mean_reprojection_error.to_bits()
        );

        // A different seed explores different minimal sets but must land on the
        // same pose to within the noise floor.
        let c = run(999);
        assert!(
            c.pose.minus(&truth).norm() < 0.01,
            "{:?}",
            c.pose.minus(&truth)
        );
        assert!(a.pose.minus(&truth).norm() < 0.01);
    }

    #[test]
    fn ransac_rejects_inputs_it_cannot_solve() {
        let k = intrinsics();
        let mut rng = DeterministicRng::new("t", 1);
        let p3 = vec![Vec3::new(0.0, 0.0, 3.0); 3];
        let p2 = vec![Vec2::new(320.0, 240.0); 3];
        assert!(solve_pnp_ransac(&p3, &p2, &k, 2.0, 100, &mut rng).is_none());
        let (a, b) = scene(&truth_pose(), &k, 10, 17);
        assert!(solve_pnp_ransac(&a, &b, &k, 2.0, 0, &mut rng).is_none());
        assert!(solve_pnp_ransac(&a, &b, &k, -1.0, 10, &mut rng).is_none());
    }

    #[test]
    fn adaptive_iteration_count_matches_the_closed_form() {
        // 50% inliers, 3-point sample, 99% confidence: log(0.01)/log(1-0.125).
        let expected = ((0.01_f64).ln() / (1.0 - 0.125_f64).ln()).ceil() as usize;
        assert_eq!(adaptive_iterations(0.5, 3, 0.99), expected);
        assert_eq!(adaptive_iterations(1.0, 3, 0.99), 1);
        assert!(adaptive_iterations(0.1, 3, 0.99) > adaptive_iterations(0.5, 3, 0.99));
    }

    #[test]
    fn a_hopeless_inlier_ratio_asks_for_an_unbounded_budget() {
        // The regression that motivates the `ln_1p` form. An eight-point model
        // whose current best hypothesis has *no* inliers must not be told that
        // one more sample will do: `w^8` underflows the `1 - x` cancellation and
        // the naive `(1.0 - w.powi(s)).ln()` returns -0.0, which a
        // "denominator is ~zero -> return 1" guard reads as "converged".
        // Every caller uses this as `adaptive_iterations(..).min(cap)`, so
        // returning 1 there silently truncates RANSAC to a single hypothesis.
        assert_eq!(adaptive_iterations(0.0, 8, 0.99), usize::MAX);
        // Monotone all the way down, with no cliff at the point where w^s stops
        // being representable as a difference from one.
        let mut previous = 0usize;
        for w in [0.9, 0.7, 0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.001] {
            let n = adaptive_iterations(w, 8, 0.99);
            assert!(n > previous, "w={w} gave {n}, not more than {previous}");
            previous = n;
        }
    }

    #[test]
    fn covariance_shrinks_as_one_over_the_point_count() {
        // Averaged over point-cloud draws so the comparison is about n, not
        // about which particular landmarks were sampled.
        let k = intrinsics();
        let truth = truth_pose();
        let (mut small, mut large) = (0.0, 0.0);
        let trials = 24;
        for seed in 0..trials {
            let (p3, _) = scene(&truth, &k, 200, 500 + seed);
            let c50 = pose_covariance(&truth, &p3[..50], &k, 1.0, None).unwrap();
            let c200 = pose_covariance(&truth, &p3, &k, 1.0, None).unwrap();
            small += c50.trace();
            large += c200.trace();
        }
        small /= trials as Scalar;
        large /= trials as Scalar;
        let ratio = small / large;
        assert!(
            (3.2..5.0).contains(&ratio),
            "4x the points should give ~4x less variance, got {ratio}"
        );
    }

    #[test]
    fn covariance_scales_with_the_square_of_the_pixel_noise() {
        let k = intrinsics();
        let truth = truth_pose();
        let (p3, _) = scene(&truth, &k, 40, 18);
        let a = pose_covariance(&truth, &p3, &k, 1.0, None).unwrap();
        let b = pose_covariance(&truth, &p3, &k, 3.0, None).unwrap();
        assert_relative_eq!(b, a * 9.0, epsilon = 1e-12);
    }

    #[test]
    fn covariance_refuses_a_degenerate_geometry() {
        let k = intrinsics();
        // Two landmarks: four equations for six unknowns.
        let p3 = vec![Vec3::new(0.0, 0.0, 3.0), Vec3::new(0.1, 0.1, 3.0)];
        assert!(pose_covariance(&Se3::identity(), &p3, &k, 1.0, None).is_none());
        // All landmarks behind the camera.
        let behind = vec![Vec3::new(0.0, 0.0, -3.0); 10];
        assert!(pose_covariance(&Se3::identity(), &behind, &k, 1.0, None).is_none());
        // Nonsense noise level.
        let (ok, _) = scene(&truth_pose(), &k, 20, 19);
        assert!(pose_covariance(&truth_pose(), &ok, &k, 0.0, None).is_none());
    }

    /// The first checkpoint of the spec.md §6 L6 claim.
    ///
    /// Over many seeded noise realisations, the normalised estimation error
    /// squared of the PnP solution against its reported covariance must average
    /// to the state dimension and land inside the chi-squared acceptance
    /// interval. An estimator that is merely *accurate* passes the tests above;
    /// only this one shows the covariance means anything.
    #[test]
    fn pnp_covariance_is_statistically_calibrated() {
        let k = intrinsics();
        let truth = truth_pose();
        let sigma = 1.0;
        let trials: usize = 300;
        let mut acc = ConsistencyAccumulator::new(6);

        for trial in 0..trials as u64 {
            // Fresh geometry every trial as well as fresh noise: a covariance
            // that is only calibrated for one lucky point cloud is not
            // calibrated.
            let (p3, clean) = scene(&truth, &k, 60, 90_000 + trial);
            let mut rng = DeterministicRng::new("pixel-noise", 700_000 + trial);
            let noisy: Vec<Vec2> = clean
                .iter()
                .map(|px| px + Vec2::new(rng.normal() * sigma, rng.normal() * sigma))
                .collect();

            // Pure least squares (Huber effectively off) so the estimator is the
            // maximum-likelihood one the covariance is derived for.
            let cfg = MotionBaConfig {
                huber_delta_px: 1e9,
                outlier_threshold_px: 1e9,
                max_iterations: 25,
                pixel_sigma: sigma,
                ..MotionBaConfig::default()
            };
            let r = refine_motion_only(&truth, &p3, &noisy, &k, None, &cfg).unwrap();
            let cov = pose_covariance(&r.pose, &p3, &k, sigma, None).unwrap();
            acc.push_pose(&r.pose, &truth, &cov);
        }

        let report = acc.report(0.05);
        assert_eq!(report.rejected, 0);
        // Report at 95%, gate at 99.99%. A per-commit test held to the 95%
        // interval fails one commit in twenty *when the estimator is correct*,
        // which is how a Tier-1 test earns a reputation for flakiness and stops
        // being read (spec.md §6). The wide interval still catches what matters:
        // a systematically overconfident covariance sits far outside it. The 95%
        // number is in the printed report for a human to look at.
        let (lo, hi) = nees_bounds(trials, 6, 1e-4);
        assert!(
            report.mean_nees >= lo && report.mean_nees <= hi,
            "{report} (gate [{lo:.3}, {hi:.3}])"
        );
        // Coverage at 300 trials has a sampling standard error of ~2.7% on the
        // 68% level alone, so the M6 exit criterion of 2% is not measurable
        // here — that number belongs to the arm rig. 10% is ~3.7 sigma.
        assert!(report.coverage_within(0.10), "{report}");
    }

    #[test]
    fn covariance_is_a_valid_covariance_matrix() {
        let k = intrinsics();
        let truth = truth_pose();
        let (p3, _) = scene(&truth, &k, 40, 20);
        let cov = pose_covariance(&truth, &p3, &k, 1.0, None).unwrap();
        assert!(wslam_core::covariance::is_valid_covariance(&cov, 1e-9));
        assert_relative_eq!(cov, cov.transpose(), epsilon = 1e-18);
        // Positive definite: every eigenvalue strictly above zero.
        assert!(cov.symmetric_eigen().eigenvalues.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn covariance_grows_when_the_landmarks_lose_depth_spread() {
        // A fronto-parallel plane leaves the pose weakly constrained along the
        // optical axis compared with a scene that has depth; the covariance must
        // say so.
        let k = intrinsics();
        let truth = Se3::identity();
        let kp = pinhole_only(&k);
        let mut flat = Vec::new();
        let mut deep = Vec::new();
        for i in 0..7 {
            for j in 0..7 {
                let x = -1.5 + 0.5 * i as Scalar;
                let y = -1.2 + 0.4 * j as Scalar;
                flat.push(Vec3::new(x, y, 5.0));
                deep.push(Vec3::new(x, y, 2.0 + 0.6 * ((i + j) % 6) as Scalar));
            }
        }
        flat.retain(|p| kp.project(p).is_some_and(|px| kp.contains(px, 0.0)));
        deep.retain(|p| kp.project(p).is_some_and(|px| kp.contains(px, 0.0)));
        let cf = pose_covariance(&truth, &flat, &k, 1.0, None).unwrap();
        let cd = pose_covariance(&truth, &deep, &k, 1.0, None).unwrap();
        assert!(
            cf[(2, 2)] > 5.0 * cd[(2, 2)],
            "flat {} vs deep {}",
            cf[(2, 2)],
            cd[(2, 2)]
        );
    }

    #[test]
    fn inlier_mask_restricts_the_covariance() {
        let k = intrinsics();
        let truth = truth_pose();
        let (p3, _) = scene(&truth, &k, 60, 21);
        let mut mask = vec![false; 60];
        mask[..20].fill(true);
        let all = pose_covariance(&truth, &p3, &k, 1.0, None).unwrap();
        let some = pose_covariance(&truth, &p3, &k, 1.0, Some(&mask)).unwrap();
        let prefix = pose_covariance(&truth, &p3[..20], &k, 1.0, None).unwrap();
        assert_relative_eq!(some, prefix, epsilon = 1e-15);
        assert!(some.trace() > all.trace());
    }

    #[test]
    fn ransac_pose_and_its_covariance_agree_with_the_noise_level() {
        // End-to-end: RANSAC on outlier-contaminated data, then the covariance
        // of the surviving inliers must bracket the truth.
        let k = intrinsics();
        let truth = truth_pose();
        let mut acc = ConsistencyAccumulator::new(6);
        let sigma = 0.8;
        let trials: usize = 120;
        for trial in 0..trials as u64 {
            let (p3, clean) = scene(&truth, &k, 80, 300_000 + trial);
            let mut rng = DeterministicRng::new("mix", 400_000 + trial);
            let noisy: Vec<Vec2> = clean
                .iter()
                .enumerate()
                .map(|(i, px)| {
                    if i % 5 == 0 {
                        Vec2::new(rng.uniform_range(0.0, 640.0), rng.uniform_range(0.0, 480.0))
                    } else {
                        px + Vec2::new(rng.normal() * sigma, rng.normal() * sigma)
                    }
                })
                .collect();
            let mut ransac = DeterministicRng::new("ransac", 500_000 + trial);
            let r = solve_pnp_ransac(&p3, &noisy, &k, 3.0 * sigma, 300, &mut ransac).unwrap();
            let cov = pose_covariance(&r.pose, &p3, &k, sigma, Some(&r.inliers)).unwrap();
            acc.push_pose(&r.pose, &truth, &cov);
        }
        let report = acc.report(0.05);
        // A robust estimator that throws away a fifth of its data is expected to
        // be slightly conservative; overconfidence is the failure that matters.
        assert!(!report.overconfident, "{report}");
        assert!(report.mean_nees < 9.0, "{report}");
        assert!(report.coverage_95 > 0.90, "{report}");
    }

    #[test]
    fn a_perturbation_of_one_sigma_gives_unit_nees_per_axis() {
        // Sanity-check the convention itself: perturb along an eigenvector of
        // the covariance by exactly one standard deviation and the NEES must be
        // one, which pins [translation; rotation] ordering and the
        // right-perturbation frame together.
        let k = intrinsics();
        let truth = truth_pose();
        let (p3, _) = scene(&truth, &k, 50, 22);
        let cov = pose_covariance(&truth, &p3, &k, 1.0, None).unwrap();
        let eig = cov.symmetric_eigen();
        for i in 0..6 {
            let dir = eig.eigenvectors.column(i).into_owned();
            let delta: Vec6 = dir * eig.eigenvalues[i].sqrt();
            let value = wslam_core::covariance::nees(&delta, &cov).unwrap();
            assert_relative_eq!(value, 1.0, epsilon = 1e-9);
        }
    }
}
