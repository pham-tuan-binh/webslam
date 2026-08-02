//! Motion-only bundle adjustment: 6-DoF pose against *fixed* 3D landmarks.
//!
//! This is the last stage of the per-frame L3 pipeline (spec.md §4 L3, "PnP
//! against the active local map"). RANSAC gives a pose that is right to within
//! the inlier threshold; this drives it to the maximum-likelihood pose and, more
//! importantly for spec.md §6 L6, produces the Hessian that the reported
//! covariance is derived from.
//!
//! ## Conventions
//!
//! - Poses are `T_world_camera`, so a landmark reaches the camera frame as
//!   `p_cam = pose.inverse() * p_world`.
//! - Perturbations are right-multiplied, `T ⊞ δ = T · exp(δ)`, and `δ` is
//!   ordered `[translation; rotation]` — the same order as the 6x6 covariance
//!   the public API promises (spec.md §3).
//! - **Pixels are undistorted.** `CameraIntrinsics::projection_jacobian` omits
//!   the distortion term deliberately, because L3 undistorts once on feature
//!   extraction and then works in the pinhole model; every function here uses
//!   [`pinhole_only`] so the residual and its Jacobian are consistent and the
//!   Jacobian is exact rather than approximate.

use nalgebra::{Matrix2x6, Matrix3x6};
use wslam_core::camera::RadialTangential;
use wslam_core::math::hat;
use wslam_core::{CameraIntrinsics, Mat3, Mat6, Scalar, Se3, Vec2, Vec3, Vec6};

/// Depth below which a landmark is treated as unobservable rather than
/// projected. The projection Jacobian divides by `z^2`, so anything near the
/// image plane produces a numerically meaningless row.
pub const MIN_DEPTH: Scalar = 1e-6;

/// Strip the distortion terms from an intrinsics.
///
/// PnP, triangulation and BA in this crate all consume *undistorted* pixel
/// coordinates; carrying the distortion coefficients into the projection would
/// silently apply the model twice and make the analytic Jacobians wrong.
#[must_use]
pub fn pinhole_only(k: &CameraIntrinsics) -> CameraIntrinsics {
    CameraIntrinsics {
        distortion: RadialTangential::NONE,
        ..*k
    }
}

/// Pinhole projection of a camera-frame point. `None` behind the image plane.
#[must_use]
pub fn project_pinhole(k: &CameraIntrinsics, p_cam: &Vec3) -> Option<Vec2> {
    if p_cam.z <= MIN_DEPTH {
        return None;
    }
    Some(Vec2::new(
        k.fx * p_cam.x / p_cam.z + k.cx,
        k.fy * p_cam.y / p_cam.z + k.cy,
    ))
}

/// Predicted pixel and the 2x6 Jacobian of that pixel with respect to a right
/// perturbation of the pose, `d(pixel) / dδ` with `T ⊞ δ = T · exp(δ)`.
///
/// Derivation, once, because getting the sign wrong here is invisible until the
/// covariance is wrong: with `p_cam = T^{-1} p_world` and `T' = T exp(δ)`,
///
/// ```text
/// p_cam' = exp(-δ) p_cam ≈ (I - hat(φ)) p_cam - ρ = p_cam - ρ + hat(p_cam) φ
/// ```
///
/// so `∂p_cam/∂ρ = -I` and `∂p_cam/∂φ = hat(p_cam)`, and the pixel Jacobian is
/// the projection Jacobian times that 3x6 block.
///
/// `k` must be pinhole; pass it through [`pinhole_only`] first.
#[must_use]
pub fn pose_jacobian(
    pose: &Se3,
    p_world: &Vec3,
    k: &CameraIntrinsics,
) -> Option<(Vec2, Matrix2x6<Scalar>)> {
    let p_cam = pose.inverse().act(p_world);
    let px = project_pinhole(k, &p_cam)?;
    let mut d_pcam = Matrix3x6::zeros();
    d_pcam
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(-Mat3::identity()));
    d_pcam.fixed_view_mut::<3, 3>(0, 3).copy_from(&hat(&p_cam));
    Some((px, k.projection_jacobian(&p_cam) * d_pcam))
}

/// Reprojection error in pixels for a single landmark, or `None` if it falls
/// behind the camera.
#[must_use]
pub fn reprojection_error(
    pose: &Se3,
    p_world: &Vec3,
    px: Vec2,
    k: &CameraIntrinsics,
) -> Option<Scalar> {
    let kp = pinhole_only(k);
    let p_cam = pose.inverse().act(p_world);
    Some((project_pinhole(&kp, &p_cam)? - px).norm())
}

/// Gauss-Newton normal equations for the reprojection problem at a pose.
#[derive(Debug, Clone, Copy)]
pub struct NormalEquations {
    /// `sum w J^T J`, the Gauss-Newton approximation to the Hessian.
    pub hessian: Mat6,
    /// `sum w J^T r`, the gradient of half the cost.
    pub gradient: Vec6,
    /// Robustified cost `sum rho(|r|)`, in squared pixels.
    pub cost: Scalar,
    /// Landmarks that contributed (in front of the camera and not masked out).
    pub used: usize,
}

/// Accumulate the normal equations at `pose`.
///
/// `mask` selects a subset (typically the RANSAC inlier set); `None` uses every
/// correspondence. `huber_delta_px` switches on the Huber M-estimator with that
/// transition point; `None` gives plain least squares, which is what the
/// covariance in spec.md §6 L6 must be derived from — a robustified Hessian is
/// not the Fisher information.
#[must_use]
pub fn normal_equations(
    pose: &Se3,
    points_3d: &[Vec3],
    points_2d: &[Vec2],
    k: &CameraIntrinsics,
    mask: Option<&[bool]>,
    huber_delta_px: Option<Scalar>,
) -> NormalEquations {
    let kp = pinhole_only(k);
    let mut out = NormalEquations {
        hessian: Mat6::zeros(),
        gradient: Vec6::zeros(),
        cost: 0.0,
        used: 0,
    };
    let n = points_3d.len().min(points_2d.len());
    for i in 0..n {
        if let Some(m) = mask {
            if !m.get(i).copied().unwrap_or(false) {
                continue;
            }
        }
        let Some((predicted, j)) = pose_jacobian(pose, &points_3d[i], &kp) else {
            continue;
        };
        let r = predicted - points_2d[i];
        let e = r.norm();
        let (w, rho) = match huber_delta_px {
            Some(d) if e > d && d > 0.0 => (d / e, d * (2.0 * e - d)),
            _ => (1.0, e * e),
        };
        out.hessian += j.transpose() * j * w;
        out.gradient += j.transpose() * r * w;
        out.cost += rho;
        out.used += 1;
    }
    out
}

/// Tuning for [`refine_motion_only`].
#[derive(Debug, Clone, Copy)]
pub struct MotionBaConfig {
    /// Maximum Levenberg-Marquardt iterations.
    pub max_iterations: usize,
    /// Huber transition point in pixels. Residuals beyond this are down-weighted
    /// linearly instead of quadratically, which is what keeps a 20% outlier
    /// fraction from dragging the pose.
    pub huber_delta_px: Scalar,
    /// Initial LM damping.
    pub initial_lambda: Scalar,
    /// Relative cost decrease below which the solve is declared converged.
    pub tolerance: Scalar,
    /// Assumed per-axis pixel measurement noise, used only for the covariance.
    pub pixel_sigma: Scalar,
    /// Reprojection error above which a landmark is reported as an outlier.
    pub outlier_threshold_px: Scalar,
}

impl Default for MotionBaConfig {
    fn default() -> Self {
        MotionBaConfig {
            max_iterations: 20,
            huber_delta_px: 2.0,
            initial_lambda: 1e-4,
            tolerance: 1e-10,
            pixel_sigma: 1.0,
            outlier_threshold_px: 4.0,
        }
    }
}

/// Result of a motion-only bundle adjustment.
#[derive(Debug, Clone)]
pub struct MotionBaResult {
    /// Refined `T_world_camera`.
    pub pose: Se3,
    /// 6x6 covariance in `[translation; rotation]`, right-perturbation, computed
    /// from the *unweighted* normal equations over the inlier set.
    pub covariance: Mat6,
    /// Per-correspondence inlier flags, full length, `false` where masked out.
    pub inliers: Vec<bool>,
    /// Number of `true` entries in [`MotionBaResult::inliers`].
    pub inlier_count: usize,
    /// Accepted LM steps.
    pub iterations: usize,
    /// Robust cost before and after each accepted step; non-increasing by
    /// construction, because a step that raises the cost is rejected.
    pub cost_history: Vec<Scalar>,
    /// Whether the relative-decrease test fired before the iteration cap.
    pub converged: bool,
    /// Mean reprojection error over the inliers, in pixels.
    pub mean_reprojection_error: Scalar,
}

/// Levenberg-Marquardt motion-only bundle adjustment.
///
/// Optimises the 6-DoF pose only; the landmarks are held fixed, which is what
/// makes this cheap enough to run every frame on the critical path.
///
/// Returns `None` if fewer than three landmarks are visible from `initial` —
/// six equations for six unknowns is already the bare minimum and anything less
/// cannot constrain the pose.
#[must_use]
pub fn refine_motion_only(
    initial: &Se3,
    points_3d: &[Vec3],
    points_2d: &[Vec2],
    k: &CameraIntrinsics,
    mask: Option<&[bool]>,
    config: &MotionBaConfig,
) -> Option<MotionBaResult> {
    let kp = pinhole_only(k);
    let huber = Some(config.huber_delta_px);

    let mut pose = *initial;
    let mut ne = normal_equations(&pose, points_3d, points_2d, &kp, mask, huber);
    if ne.used < 3 {
        return None;
    }
    let mut cost = ne.cost;
    let mut lambda = config.initial_lambda;
    let mut history = vec![cost];
    let mut accepted = 0usize;
    let mut converged = false;

    for _ in 0..config.max_iterations {
        let mut stepped = false;
        // Inner loop escalates damping until a step actually reduces the cost.
        // Bounded at 12 escalations: 1e-4 * 10^12 is a gradient step so small
        // that no further progress is possible in f64.
        for _ in 0..12 {
            // Marquardt scaling (damp proportional to the diagonal) rather than
            // Levenberg's `lambda * I`: the translation and rotation blocks of
            // this Hessian differ by orders of magnitude in units, and a uniform
            // damping term would freeze one of them.
            let mut damped = ne.hessian;
            for d in 0..6 {
                let diag = ne.hessian[(d, d)];
                damped[(d, d)] += lambda * if diag > 0.0 { diag } else { 1.0 };
            }
            let Some(step) = solve_spd(&damped, &(-ne.gradient)) else {
                lambda *= 10.0;
                continue;
            };
            let candidate = pose.plus(&step);
            let cand_ne = normal_equations(&candidate, points_3d, points_2d, &kp, mask, huber);
            if cand_ne.used >= 3 && cand_ne.cost < cost {
                pose = candidate;
                let previous = cost;
                cost = cand_ne.cost;
                ne = cand_ne;
                history.push(cost);
                accepted += 1;
                lambda = (lambda * 0.3).max(1e-12);
                stepped = true;
                converged = previous - cost <= config.tolerance * (1.0 + previous);
                break;
            }
            lambda *= 10.0;
        }
        if !stepped {
            // No damping level helped: we are at a minimum to within f64.
            converged = true;
            break;
        }
        if converged {
            break;
        }
    }

    // Classify inliers against the refined pose, then take the covariance from
    // the unweighted normal equations over that set.
    let n = points_3d.len().min(points_2d.len());
    let mut inliers = vec![false; points_3d.len()];
    let mut err_sum = 0.0;
    let mut inlier_count = 0usize;
    for (i, flag) in inliers.iter_mut().enumerate().take(n) {
        if let Some(m) = mask {
            if !m.get(i).copied().unwrap_or(false) {
                continue;
            }
        }
        let Some(e) = reprojection_error(&pose, &points_3d[i], points_2d[i], &kp) else {
            continue;
        };
        if e <= config.outlier_threshold_px {
            *flag = true;
            inlier_count += 1;
            err_sum += e;
        }
    }

    let covariance =
        crate::pnp::pose_covariance(&pose, points_3d, &kp, config.pixel_sigma, Some(&inliers))
            .unwrap_or_else(Mat6::zeros);

    Some(MotionBaResult {
        pose,
        covariance,
        inliers,
        inlier_count,
        iterations: accepted,
        cost_history: history,
        converged,
        mean_reprojection_error: if inlier_count > 0 {
            err_sum / inlier_count as Scalar
        } else {
            Scalar::INFINITY
        },
    })
}

/// Solve `A x = b` for a symmetric positive-definite `A`, falling back to LU if
/// the Cholesky factorisation fails (which is how a rank-deficient geometry
/// announces itself).
fn solve_spd(a: &Mat6, b: &Vec6) -> Option<Vec6> {
    let x = match nalgebra::linalg::Cholesky::new(*a) {
        Some(chol) => chol.solve(b),
        None => a.lu().solve(b)?,
    };
    x.iter().all(|v| v.is_finite()).then_some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use wslam_core::{DeterministicRng, So3};

    fn intrinsics() -> CameraIntrinsics {
        CameraIntrinsics::from_focal(520.0, 640, 480)
    }

    fn truth_pose() -> Se3 {
        Se3::new(
            So3::exp(&Vec3::new(0.06, -0.11, 0.04)),
            Vec3::new(0.35, -0.2, -2.5),
        )
    }

    /// Landmarks on a deterministic jittered grid, kept only where they project
    /// inside the image from `pose`.
    fn scene(pose: &Se3, k: &CameraIntrinsics, n: usize, seed: u64) -> (Vec<Vec3>, Vec<Vec2>) {
        let mut rng = DeterministicRng::new("scene", seed);
        let mut p3 = Vec::new();
        let mut p2 = Vec::new();
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
            match project_pinhole(k, &cam) {
                Some(px) if k.contains(px, 4.0) => {
                    p3.push(p);
                    p2.push(px);
                }
                _ => {}
            }
        }
        (p3, p2)
    }

    #[test]
    fn pose_jacobian_matches_central_differences() {
        let k = intrinsics();
        let pose = truth_pose();
        let p = Vec3::new(0.7, -0.4, 3.2);
        let (_, j) = pose_jacobian(&pose, &p, &k).unwrap();
        let eps = 1e-7;
        for i in 0..6 {
            let mut d = Vec6::zeros();
            d[i] = eps;
            let fwd = project_pinhole(&k, &pose.plus(&d).inverse().act(&p)).unwrap();
            let bwd = project_pinhole(&k, &pose.plus(&(-d)).inverse().act(&p)).unwrap();
            let num = (fwd - bwd) / (2.0 * eps);
            assert_relative_eq!(j[(0, i)], num.x, epsilon = 1e-4, max_relative = 1e-5);
            assert_relative_eq!(j[(1, i)], num.y, epsilon = 1e-4, max_relative = 1e-5);
        }
    }

    #[test]
    fn pose_jacobian_rejects_landmarks_behind_the_camera() {
        let k = intrinsics();
        let pose = Se3::identity();
        assert!(pose_jacobian(&pose, &Vec3::new(0.1, 0.1, -2.0), &k).is_none());
        assert!(pose_jacobian(&pose, &Vec3::new(0.1, 0.1, 0.0), &k).is_none());
    }

    #[test]
    fn gradient_vanishes_at_the_noise_free_solution() {
        let k = intrinsics();
        let truth = truth_pose();
        let (p3, p2) = scene(&truth, &k, 40, 1);
        let ne = normal_equations(&truth, &p3, &p2, &k, None, None);
        assert_eq!(ne.used, 40);
        assert!(ne.cost < 1e-18, "cost {}", ne.cost);
        assert!(ne.gradient.norm() < 1e-9, "grad {}", ne.gradient.norm());
    }

    #[test]
    fn converges_back_to_truth_from_a_perturbed_start() {
        let k = intrinsics();
        let truth = truth_pose();
        let (p3, p2) = scene(&truth, &k, 60, 2);
        let start = truth.plus(&Vec6::new(0.25, -0.18, 0.3, 0.05, -0.06, 0.04));
        let r = refine_motion_only(&start, &p3, &p2, &k, None, &MotionBaConfig::default()).unwrap();
        let err = r.pose.minus(&truth);
        assert!(err.norm() < 1e-8, "residual twist {err:?}");
        assert_eq!(r.inlier_count, 60);
        assert!(r.converged);
    }

    #[test]
    fn cost_history_is_monotonically_non_increasing() {
        let k = intrinsics();
        let truth = truth_pose();
        let (p3, p2) = scene(&truth, &k, 50, 3);
        let start = truth.plus(&Vec6::new(-0.4, 0.3, 0.5, 0.09, 0.07, -0.08));
        let r = refine_motion_only(&start, &p3, &p2, &k, None, &MotionBaConfig::default()).unwrap();
        assert!(r.cost_history.len() >= 2, "no steps taken");
        for w in r.cost_history.windows(2) {
            assert!(w[1] <= w[0], "cost rose: {:?}", r.cost_history);
        }
        assert!(*r.cost_history.last().unwrap() < 1e-12 * r.cost_history[0].max(1.0));
    }

    #[test]
    fn huber_keeps_the_pose_near_truth_with_twenty_percent_outliers() {
        let k = intrinsics();
        let truth = truth_pose();
        let (p3, mut p2) = scene(&truth, &k, 100, 4);
        let mut rng = DeterministicRng::new("outliers", 77);
        // Every fifth measurement is gross: displaced by 40-120 px in a random
        // direction, which is what a mistracked KLT feature looks like.
        let mut corrupted = Vec::new();
        for (i, px) in p2.iter_mut().enumerate() {
            if i % 5 == 0 {
                let angle = rng.uniform_range(0.0, std::f64::consts::TAU);
                let mag = rng.uniform_range(40.0, 120.0);
                *px += Vec2::new(mag * angle.cos(), mag * angle.sin());
                corrupted.push(i);
            }
        }
        let start = truth.plus(&Vec6::new(0.08, -0.05, 0.1, 0.02, -0.02, 0.01));
        let cfg = MotionBaConfig {
            huber_delta_px: 2.0,
            outlier_threshold_px: 5.0,
            max_iterations: 40,
            ..MotionBaConfig::default()
        };
        let r = refine_motion_only(&start, &p3, &p2, &k, None, &cfg).unwrap();

        let err = r.pose.minus(&truth);
        let (rho, phi) = wslam_core::math::split_twist(&err);
        assert!(rho.norm() < 0.02, "translation drifted {}", rho.norm());
        assert!(phi.norm() < 0.005, "rotation drifted {}", phi.norm());
        // Every corrupted measurement must be flagged, and the clean ones kept.
        for i in corrupted {
            assert!(!r.inliers[i], "outlier {i} accepted");
        }
        assert_eq!(r.inlier_count, 80);
    }

    #[test]
    fn masked_correspondences_are_ignored_entirely() {
        let k = intrinsics();
        let truth = truth_pose();
        let (p3, mut p2) = scene(&truth, &k, 30, 5);
        let mut mask = vec![true; 30];
        // Wreck the last ten but mask them out; the fit must be unaffected.
        for (i, px) in p2.iter_mut().enumerate().skip(20) {
            *px += Vec2::new(300.0, -250.0);
            mask[i] = false;
        }
        let r = refine_motion_only(
            &truth.plus(&Vec6::new(0.1, 0.1, 0.1, 0.01, 0.01, 0.01)),
            &p3,
            &p2,
            &k,
            Some(&mask),
            &MotionBaConfig::default(),
        )
        .unwrap();
        assert!(r.pose.minus(&truth).norm() < 1e-8);
        assert_eq!(r.inlier_count, 20);
        assert!(r.inliers[..20].iter().all(|&b| b));
        assert!(r.inliers[20..].iter().all(|&b| !b));
    }

    #[test]
    fn rejects_a_scene_it_cannot_see() {
        let k = intrinsics();
        // Landmarks all behind the camera.
        let p3 = vec![Vec3::new(0.0, 0.0, -5.0); 10];
        let p2 = vec![Vec2::new(320.0, 240.0); 10];
        assert!(refine_motion_only(
            &Se3::identity(),
            &p3,
            &p2,
            &k,
            None,
            &MotionBaConfig::default()
        )
        .is_none());
    }

    #[test]
    fn huber_weighting_lowers_the_cost_of_a_gross_residual() {
        let k = intrinsics();
        let truth = truth_pose();
        let (p3, mut p2) = scene(&truth, &k, 10, 6);
        p2[0] += Vec2::new(100.0, 0.0);
        let plain = normal_equations(&truth, &p3, &p2, &k, None, None);
        let robust = normal_equations(&truth, &p3, &p2, &k, None, Some(2.0));
        // 100 px: quadratic cost 1e4, Huber cost 2*2*100 - 4 = 396.
        assert_relative_eq!(plain.cost, 1e4, epsilon = 1e-6);
        assert_relative_eq!(robust.cost, 396.0, epsilon = 1e-6);
    }
}
