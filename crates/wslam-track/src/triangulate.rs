//! Triangulation of a landmark from two or more calibrated views.
//!
//! The gate on this module is the parallax check. A landmark observed twice from
//! nearly the same place has a depth that is not merely uncertain but
//! *unbounded*: the information matrix approaches rank 2 and the covariance
//! along the optical axis diverges. Accepting such a point and reporting a large
//! covariance is not honest either, because the linearisation the covariance
//! comes from stopped being valid long before the number got large. So this
//! module **rejects** it, and says why — spec.md §6 L6 is only defensible if the
//! points feeding it were themselves observable.
//!
//! All 2D inputs are **undistorted** pixel coordinates; see the module docs of
//! [`crate::motion_ba`].

use nalgebra::DMatrix;
use wslam_core::{CameraIntrinsics, Error, Mat3, Scalar, Se3, Vec2, Vec3};

use crate::motion_ba::{pinhole_only, project_pinhole};

/// Why a candidate landmark was not accepted.
///
/// Distinguished rather than collapsed into `None` because the frontend tunes
/// on the mix: mostly `LowParallax` means the camera is not translating, mostly
/// `HighReprojection` means the correspondences are wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriangulationRejection {
    /// Fewer than two views were supplied.
    TooFewViews,
    /// The linear system was rank deficient, or the solution is a point at
    /// infinity (homogeneous coordinate indistinguishable from zero).
    Degenerate,
    /// The point falls behind at least one camera.
    Cheirality,
    /// The viewing rays are too close to parallel for the depth to be observable.
    LowParallax,
    /// The point is further away than the configured horizon.
    TooFar,
    /// The point does not reproject onto its own observations.
    HighReprojection,
}

impl std::fmt::Display for TriangulationRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            TriangulationRejection::TooFewViews => "fewer than two views",
            TriangulationRejection::Degenerate => "rank-deficient linear system",
            TriangulationRejection::Cheirality => "point behind a camera",
            TriangulationRejection::LowParallax => "insufficient parallax",
            TriangulationRejection::TooFar => "beyond the depth horizon",
            TriangulationRejection::HighReprojection => "reprojection error too large",
        };
        f.write_str(s)
    }
}

impl From<TriangulationRejection> for Error {
    fn from(r: TriangulationRejection) -> Error {
        // Every rejection is "keep feeding frames", never a hard failure: more
        // baseline or a better match fixes all six.
        Error::insufficient(format!("triangulation: {r}"))
    }
}

/// Acceptance criteria for a triangulated landmark.
#[derive(Debug, Clone, Copy)]
pub struct TriangulationConfig {
    /// Minimum angle subtended at the landmark by the two most separated
    /// viewpoints, in radians.
    pub min_parallax_rad: Scalar,
    /// Maximum per-view reprojection error, in pixels.
    pub max_reprojection_px: Scalar,
    /// Per-axis pixel measurement noise, used for the reported covariance.
    pub pixel_sigma: Scalar,
    /// Depth horizon in world units. Landmarks past this are treated as
    /// direction-only and dropped.
    pub max_depth: Scalar,
}

impl Default for TriangulationConfig {
    fn default() -> Self {
        TriangulationConfig {
            // One degree at 5 m is a 9 cm baseline: below that, a one-pixel
            // matching error moves the depth by more than a metre.
            min_parallax_rad: 1.0_f64.to_radians(),
            max_reprojection_px: 4.0,
            pixel_sigma: 1.0,
            max_depth: 1.0e4,
        }
    }
}

/// An accepted landmark and everything the caller needs to judge it.
#[derive(Debug, Clone, Copy)]
pub struct TriangulatedPoint {
    /// Position in world coordinates.
    pub position: Vec3,
    /// 3x3 position covariance, `pixel_sigma^2 (J^T J)^-1`.
    pub covariance: Mat3,
    /// Largest angle subtended at the point by any pair of viewpoints.
    pub parallax_rad: Scalar,
    /// Worst per-view reprojection error, in pixels.
    pub max_reprojection_px: Scalar,
    /// Views that contributed.
    pub views: usize,
}

/// Linear DLT triangulation from two views. Returns the raw solution with no
/// checks applied; prefer [`triangulate_two_view`].
#[must_use]
pub fn triangulate_dlt(
    pose_a: &Se3,
    px_a: Vec2,
    pose_b: &Se3,
    px_b: Vec2,
    k: &CameraIntrinsics,
) -> Option<Vec3> {
    triangulate_dlt_n(&[(*pose_a, px_a), (*pose_b, px_b)], k)
}

/// Linear DLT triangulation from n views.
///
/// Each observation contributes the two independent rows of `x × (P X) = 0` in
/// normalised image coordinates, and the solution is the right singular vector
/// of the smallest singular value. Working in normalised rather than pixel
/// coordinates is what keeps the 2n x 4 system conditioned; in pixels the rows
/// differ in magnitude by the focal length.
#[must_use]
pub fn triangulate_dlt_n(observations: &[(Se3, Vec2)], k: &CameraIntrinsics) -> Option<Vec3> {
    if observations.len() < 2 {
        return None;
    }
    let kp = pinhole_only(k);
    let mut a = DMatrix::<Scalar>::zeros(2 * observations.len(), 4);
    for (i, (pose, px)) in observations.iter().enumerate() {
        let n = kp.unproject_normalized(*px);
        let t_cw = pose.inverse();
        let r = t_cw.rotation().matrix();
        let t = t_cw.translation();
        let row0 = r.row(2) * n.x - r.row(0);
        let row1 = r.row(2) * n.y - r.row(1);
        a.fixed_view_mut::<1, 3>(2 * i, 0).copy_from(&row0);
        a.fixed_view_mut::<1, 3>(2 * i + 1, 0).copy_from(&row1);
        a[(2 * i, 3)] = n.x * t.z - t.x;
        a[(2 * i + 1, 3)] = n.y * t.z - t.y;
    }

    let svd = a.svd(false, true);
    let v_t = svd.v_t?;
    let idx = svd
        .singular_values
        .iter()
        .enumerate()
        .min_by(|x, y| x.1.partial_cmp(y.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)?;
    let v = v_t.row(idx);
    // v is unit norm, so a homogeneous coordinate this small is a point at
    // infinity rather than a scaling artefact.
    if v[3].abs() < 1e-12 {
        return None;
    }
    let p = Vec3::new(v[0] / v[3], v[1] / v[3], v[2] / v[3]);
    p.iter().all(|c| c.is_finite()).then_some(p)
}

/// Midpoint triangulation: the world point minimising the sum of squared
/// distances to the viewing rays.
///
/// Closed form — `sum (I - d d^T)` is the normal matrix and `sum (I - d d^T) o`
/// the right-hand side. Cheaper than the DLT and a useful independent check,
/// but it minimises a geometric distance in world units rather than a
/// reprojection error, so it is not the estimator to report.
#[must_use]
pub fn triangulate_midpoint(observations: &[(Se3, Vec2)], k: &CameraIntrinsics) -> Option<Vec3> {
    if observations.len() < 2 {
        return None;
    }
    let kp = pinhole_only(k);
    let mut normal = Mat3::zeros();
    let mut rhs = Vec3::zeros();
    for (pose, px) in observations {
        let d = pose.rotation().act(&kp.unproject_bearing(*px));
        let o = pose.translation();
        let projector = Mat3::identity() - d * d.transpose();
        normal += projector;
        rhs += projector * o;
    }
    let x = normal.try_inverse()? * rhs;
    x.iter().all(|c| c.is_finite()).then_some(x)
}

/// Largest angle subtended at `point` by any pair of viewpoints.
///
/// Measured between the rays from the *camera centres* to the estimated point
/// rather than between the measured bearings, so that the number reported is
/// the geometric parallax of the solution rather than of the raw observations.
#[must_use]
pub fn max_parallax(point: &Vec3, poses: &[Se3]) -> Scalar {
    let dirs: Vec<Vec3> = poses
        .iter()
        .map(|p| point - p.translation())
        .filter(|v| v.norm() > 1e-12)
        .map(|v| v.normalize())
        .collect();
    let mut best = 0.0_f64;
    for (i, a) in dirs.iter().enumerate() {
        for b in dirs.iter().skip(i + 1) {
            best = best.max(a.dot(b).clamp(-1.0, 1.0).acos());
        }
    }
    best
}

/// 3x3 position covariance of a triangulated landmark.
///
/// `pixel_sigma^2 (sum J_i^T J_i)^-1` with `J_i = dpi/dp_cam * R_i^T`, the
/// same Gauss-Newton information as [`crate::pnp::pose_covariance`] but
/// differentiated with respect to the point instead of the pose.
///
/// `None` when fewer than two views see the point or the information matrix is
/// singular — which is precisely the low-parallax case this module refuses.
#[must_use]
pub fn point_covariance(
    point: &Vec3,
    poses: &[Se3],
    k: &CameraIntrinsics,
    pixel_sigma: Scalar,
) -> Option<Mat3> {
    if !(pixel_sigma.is_finite() && pixel_sigma > 0.0) {
        return None;
    }
    let kp = pinhole_only(k);
    let mut information = Mat3::zeros();
    let mut used = 0usize;
    for pose in poses {
        let t_cw = pose.inverse();
        let p_cam = t_cw.act(point);
        if p_cam.z <= crate::motion_ba::MIN_DEPTH {
            continue;
        }
        // p_cam = R_cw (X - t_wc)  =>  dp_cam/dX = R_cw
        let j = kp.projection_jacobian(&p_cam) * t_cw.rotation().matrix();
        information += j.transpose() * j;
        used += 1;
    }
    if used < 2 {
        return None;
    }
    // Symmetrise by hand: `wslam_core::covariance::symmetrize` is 6x6 only, and
    // the inverse of an accumulated outer-product sum drifts out of symmetry.
    let raw = information.try_inverse()? * (pixel_sigma * pixel_sigma);
    let cov = (raw + raw.transpose()) * 0.5;
    let sane = cov.iter().all(|v| v.is_finite()) && (0..3).all(|i| cov[(i, i)] > 0.0);
    sane.then_some(cov)
}

/// Triangulate from two views with every check applied.
pub fn triangulate_two_view(
    pose_a: &Se3,
    px_a: Vec2,
    pose_b: &Se3,
    px_b: Vec2,
    k: &CameraIntrinsics,
    config: &TriangulationConfig,
) -> Result<TriangulatedPoint, TriangulationRejection> {
    triangulate_n_view(&[(*pose_a, px_a), (*pose_b, px_b)], k, config)
}

/// Triangulate from n views with every check applied.
///
/// Check order is deliberate: cheirality first because a point behind a camera
/// makes the parallax and reprojection numbers meaningless, then parallax,
/// then the depth horizon, then reprojection.
pub fn triangulate_n_view(
    observations: &[(Se3, Vec2)],
    k: &CameraIntrinsics,
    config: &TriangulationConfig,
) -> Result<TriangulatedPoint, TriangulationRejection> {
    if observations.len() < 2 {
        return Err(TriangulationRejection::TooFewViews);
    }
    let kp = pinhole_only(k);
    let position =
        triangulate_dlt_n(observations, &kp).ok_or(TriangulationRejection::Degenerate)?;

    let mut worst_reprojection = 0.0_f64;
    for (pose, px) in observations {
        let p_cam = pose.inverse().act(&position);
        let predicted = project_pinhole(&kp, &p_cam).ok_or(TriangulationRejection::Cheirality)?;
        worst_reprojection = worst_reprojection.max((predicted - px).norm());
        if p_cam.z > config.max_depth {
            return Err(TriangulationRejection::TooFar);
        }
    }

    let poses: Vec<Se3> = observations.iter().map(|(p, _)| *p).collect();
    let parallax = max_parallax(&position, &poses);
    if !(parallax.is_finite() && parallax >= config.min_parallax_rad) {
        return Err(TriangulationRejection::LowParallax);
    }
    if worst_reprojection > config.max_reprojection_px {
        return Err(TriangulationRejection::HighReprojection);
    }

    let covariance = point_covariance(&position, &poses, &kp, config.pixel_sigma)
        .ok_or(TriangulationRejection::Degenerate)?;

    Ok(TriangulatedPoint {
        position,
        covariance,
        parallax_rad: parallax,
        max_reprojection_px: worst_reprojection,
        views: observations.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use wslam_core::{DeterministicRng, So3};

    fn intrinsics() -> CameraIntrinsics {
        CameraIntrinsics::from_focal(520.0, 640, 480)
    }

    /// A camera at `centre` looking straight down world +Z.
    fn cam_at(x: Scalar, y: Scalar, z: Scalar) -> Se3 {
        Se3::from_translation(Vec3::new(x, y, z))
    }

    fn observe(pose: &Se3, point: &Vec3, k: &CameraIntrinsics) -> Vec2 {
        project_pinhole(&pinhole_only(k), &pose.inverse().act(point)).unwrap()
    }

    /// Projection without the depth guard, so a test can synthesise the
    /// observation a camera *would* report for a point behind it. Real cameras
    /// cannot, but a mistracked correspondence produces exactly this geometry.
    fn observe_signed(pose: &Se3, point: &Vec3, k: &CameraIntrinsics) -> Vec2 {
        let p = pose.inverse().act(point);
        Vec2::new(k.fx * p.x / p.z + k.cx, k.fy * p.y / p.z + k.cy)
    }

    #[test]
    fn recovers_a_known_point_exactly() {
        let k = intrinsics();
        let a = cam_at(0.0, 0.0, 0.0);
        let b = Se3::new(
            So3::exp(&Vec3::new(0.0, -0.08, 0.0)),
            Vec3::new(0.4, 0.05, 0.1),
        );
        let truth = Vec3::new(0.3, -0.2, 4.0);
        let p = triangulate_two_view(
            &a,
            observe(&a, &truth, &k),
            &b,
            observe(&b, &truth, &k),
            &k,
            &TriangulationConfig::default(),
        )
        .unwrap();
        assert_relative_eq!(p.position, truth, epsilon = 1e-10);
        assert!(p.max_reprojection_px < 1e-9);
        assert_eq!(p.views, 2);
    }

    #[test]
    fn midpoint_and_dlt_agree_on_noise_free_data() {
        let k = intrinsics();
        let a = cam_at(0.0, 0.0, 0.0);
        let b = cam_at(0.5, 0.0, 0.0);
        let truth = Vec3::new(-0.1, 0.3, 5.0);
        let obs = [(a, observe(&a, &truth, &k)), (b, observe(&b, &truth, &k))];
        assert_relative_eq!(triangulate_dlt_n(&obs, &k).unwrap(), truth, epsilon = 1e-10);
        assert_relative_eq!(
            triangulate_midpoint(&obs, &k).unwrap(),
            truth,
            epsilon = 1e-10
        );
    }

    #[test]
    fn n_view_beats_two_view_under_noise() {
        let k = intrinsics();
        let truth = Vec3::new(0.2, -0.1, 6.0);
        let mut rng = DeterministicRng::new("tri-noise", 5);
        let mut two_err = 0.0;
        let mut many_err = 0.0;
        for _ in 0..200 {
            let mut obs = Vec::new();
            for i in 0..8 {
                let pose = cam_at(-0.7 + 0.2 * i as Scalar, 0.0, 0.0);
                let px =
                    observe(&pose, &truth, &k) + Vec2::new(rng.normal() * 0.5, rng.normal() * 0.5);
                obs.push((pose, px));
            }
            let two = triangulate_dlt_n(&[obs[0], obs[7]], &k).unwrap();
            let many = triangulate_dlt_n(&obs, &k).unwrap();
            two_err += (two - truth).norm();
            many_err += (many - truth).norm();
        }
        assert!(
            many_err < two_err,
            "eight views {many_err} should beat two {two_err}"
        );
    }

    #[test]
    fn near_zero_parallax_is_rejected_not_returned_with_a_huge_number() {
        let k = intrinsics();
        // 1 mm of baseline, landmark at 5 m: 0.011 degrees of parallax. The
        // point is in front of both cameras and reprojects perfectly, so only
        // the parallax test can catch it.
        let a = cam_at(0.0, 0.0, 0.0);
        let b = cam_at(0.001, 0.0, 0.0);
        let truth = Vec3::new(0.0, 0.0, 5.0);
        let err = triangulate_two_view(
            &a,
            observe(&a, &truth, &k),
            &b,
            observe(&b, &truth, &k),
            &k,
            &TriangulationConfig::default(),
        )
        .unwrap_err();
        assert_eq!(err, TriangulationRejection::LowParallax);
    }

    #[test]
    fn pure_rotation_is_rejected() {
        // The tier-2 survival case (spec.md §6 L3): the user pans without
        // translating. There is no baseline at all, so nothing is observable.
        let k = intrinsics();
        let a = Se3::identity();
        let b = Se3::from_rotation(So3::exp(&Vec3::new(0.0, 0.12, 0.0)));
        let truth = Vec3::new(0.1, 0.0, 4.0);
        let result = triangulate_two_view(
            &a,
            observe(&a, &truth, &k),
            &b,
            observe(&b, &truth, &k),
            &k,
            &TriangulationConfig::default(),
        );
        assert!(result.is_err(), "pure rotation must not triangulate");
    }

    #[test]
    fn points_behind_the_camera_are_rejected() {
        let k = intrinsics();
        let a = cam_at(0.0, 0.0, 0.0);
        let b = cam_at(0.6, 0.0, 0.0);
        // A landmark behind both cameras. The two rays still intersect and the
        // DLT still has a unique answer, so only the sign of the depth
        // distinguishes this from a valid landmark — which is exactly the trap
        // `CameraIntrinsics::project` warns about.
        let behind = Vec3::new(0.3, 0.1, -4.0);
        let err = triangulate_two_view(
            &a,
            observe_signed(&a, &behind, &k),
            &b,
            observe_signed(&b, &behind, &k),
            &k,
            &TriangulationConfig::default(),
        )
        .unwrap_err();
        assert_eq!(err, TriangulationRejection::Cheirality);
        // And the DLT on its own really did find the point, so the rejection is
        // the check firing rather than the solve failing.
        let raw = triangulate_dlt(
            &a,
            observe_signed(&a, &behind, &k),
            &b,
            observe_signed(&b, &behind, &k),
            &k,
        )
        .unwrap();
        assert_relative_eq!(raw, behind, epsilon = 1e-10);
    }

    #[test]
    fn a_point_behind_one_camera_only_is_rejected() {
        let k = intrinsics();
        // Camera b sits beyond the landmark and faces back along -Z, so a point
        // further out than b is in front of a and behind b.
        let a = Se3::identity();
        let b = Se3::new(
            So3::exp(&Vec3::new(0.0, std::f64::consts::PI, 0.0)),
            Vec3::new(0.0, 0.0, 8.0),
        );
        let point = Vec3::new(0.2, 0.0, 10.0);
        assert!(a.inverse().act(&point).z > 0.0, "must be in front of a");
        assert!(b.inverse().act(&point).z < 0.0, "must be behind b");
        let err = triangulate_two_view(
            &a,
            observe_signed(&a, &point, &k),
            &b,
            observe_signed(&b, &point, &k),
            &k,
            &TriangulationConfig::default(),
        )
        .unwrap_err();
        assert_eq!(err, TriangulationRejection::Cheirality);
    }

    #[test]
    fn a_bad_correspondence_is_rejected_on_reprojection() {
        let k = intrinsics();
        let a = cam_at(0.0, 0.0, 0.0);
        let b = cam_at(0.8, 0.0, 0.0);
        let truth = Vec3::new(0.1, 0.2, 4.0);
        let px_a = observe(&a, &truth, &k);
        // Shift only the y coordinate: the epipolar geometry is horizontal here,
        // so a vertical error cannot be absorbed by any depth.
        let px_b = observe(&b, &truth, &k) + Vec2::new(0.0, 30.0);
        let err = triangulate_two_view(&a, px_a, &b, px_b, &k, &TriangulationConfig::default())
            .unwrap_err();
        assert_eq!(err, TriangulationRejection::HighReprojection);
    }

    #[test]
    fn the_depth_horizon_is_enforced() {
        let k = intrinsics();
        let a = cam_at(0.0, 0.0, 0.0);
        let b = cam_at(200.0, 0.0, 0.0);
        let truth = Vec3::new(0.0, 0.0, 5000.0);
        let cfg = TriangulationConfig {
            max_depth: 100.0,
            ..TriangulationConfig::default()
        };
        let err = triangulate_two_view(
            &a,
            observe(&a, &truth, &k),
            &b,
            observe(&b, &truth, &k),
            &k,
            &cfg,
        )
        .unwrap_err();
        assert_eq!(err, TriangulationRejection::TooFar);
    }

    #[test]
    fn too_few_views_is_reported_as_such() {
        let k = intrinsics();
        assert_eq!(
            triangulate_n_view(&[], &k, &TriangulationConfig::default()).unwrap_err(),
            TriangulationRejection::TooFewViews
        );
        assert!(triangulate_dlt_n(&[(Se3::identity(), Vec2::zeros())], &k).is_none());
        assert!(triangulate_midpoint(&[(Se3::identity(), Vec2::zeros())], &k).is_none());
    }

    #[test]
    fn parallax_matches_the_closed_form_angle() {
        // Two cameras 2 m apart, point 1 m in front of the midpoint: the rays
        // meet at exactly 90 degrees.
        let point = Vec3::new(0.0, 0.0, 1.0);
        let poses = [cam_at(-1.0, 0.0, 0.0), cam_at(1.0, 0.0, 0.0)];
        assert_relative_eq!(
            max_parallax(&point, &poses),
            std::f64::consts::FRAC_PI_2,
            epsilon = 1e-12
        );
    }

    #[test]
    fn parallax_is_the_maximum_over_all_pairs() {
        let point = Vec3::new(0.0, 0.0, 1.0);
        let poses = [
            cam_at(-1.0, 0.0, 0.0),
            cam_at(0.0, 0.0, 0.0),
            cam_at(1.0, 0.0, 0.0),
        ];
        assert_relative_eq!(
            max_parallax(&point, &poses),
            std::f64::consts::FRAC_PI_2,
            epsilon = 1e-12
        );
    }

    #[test]
    fn covariance_elongates_along_the_ray_as_the_baseline_shrinks() {
        let k = intrinsics();
        let truth = Vec3::new(0.0, 0.0, 5.0);
        let wide = [cam_at(-0.5, 0.0, 0.0), cam_at(0.5, 0.0, 0.0)];
        let narrow = [cam_at(-0.05, 0.0, 0.0), cam_at(0.05, 0.0, 0.0)];
        let cw = point_covariance(&truth, &wide, &k, 1.0).unwrap();
        let cn = point_covariance(&truth, &narrow, &k, 1.0).unwrap();
        // Depth variance grows as the inverse square of the baseline, so a 10x
        // shorter baseline is ~100x worse; lateral variance barely moves.
        let depth_ratio = cn[(2, 2)] / cw[(2, 2)];
        let lateral_ratio = cn[(1, 1)] / cw[(1, 1)];
        assert!(
            (60.0..160.0).contains(&depth_ratio),
            "depth ratio {depth_ratio}"
        );
        assert!(lateral_ratio < 2.0, "lateral ratio {lateral_ratio}");
        assert!(
            cn[(2, 2)] > 50.0 * cn[(1, 1)],
            "narrow baseline must be depth-dominated"
        );
    }

    #[test]
    fn covariance_scales_with_the_square_of_the_pixel_noise() {
        let k = intrinsics();
        let truth = Vec3::new(0.1, 0.1, 4.0);
        let poses = [cam_at(-0.4, 0.0, 0.0), cam_at(0.4, 0.0, 0.0)];
        let a = point_covariance(&truth, &poses, &k, 1.0).unwrap();
        let b = point_covariance(&truth, &poses, &k, 2.5).unwrap();
        assert_relative_eq!(b, a * 6.25, epsilon = 1e-12);
    }

    #[test]
    fn covariance_rejects_a_single_view() {
        let k = intrinsics();
        let poses = [cam_at(0.0, 0.0, 0.0)];
        assert!(point_covariance(&Vec3::new(0.0, 0.0, 3.0), &poses, &k, 1.0).is_none());
        // Zero sigma is not a covariance.
        let two = [cam_at(-0.3, 0.0, 0.0), cam_at(0.3, 0.0, 0.0)];
        assert!(point_covariance(&Vec3::new(0.0, 0.0, 3.0), &two, &k, 0.0).is_none());
    }

    /// The triangulated point's error must be consistent with the covariance it
    /// reports — the point-level analogue of the pose NEES in `pnp`.
    #[test]
    fn reported_point_covariance_is_calibrated() {
        let k = intrinsics();
        let truth = Vec3::new(0.15, -0.1, 4.0);
        let poses = [
            cam_at(-0.6, 0.0, 0.0),
            cam_at(0.0, 0.1, 0.0),
            cam_at(0.6, -0.1, 0.05),
        ];
        let sigma = 0.7;
        let mut rng = DeterministicRng::new("tri-nees", 20260801);
        let cov = point_covariance(&truth, &poses, &k, sigma).unwrap();
        let inv = cov.try_inverse().unwrap();

        let trials = 2000;
        let mut total = 0.0;
        for _ in 0..trials {
            let obs: Vec<(Se3, Vec2)> = poses
                .iter()
                .map(|p| {
                    (
                        *p,
                        observe(p, &truth, &k)
                            + Vec2::new(rng.normal() * sigma, rng.normal() * sigma),
                    )
                })
                .collect();
            let est = triangulate_dlt_n(&obs, &k).unwrap();
            let e = est - truth;
            total += (e.transpose() * inv * e)[(0, 0)];
        }
        let mean = total / trials as Scalar;
        let (lo, hi) = wslam_core::stats::nees_bounds(trials, 3, 0.01);
        assert!(
            mean > lo && mean < hi,
            "point NEES {mean} outside [{lo:.3}, {hi:.3}]"
        );
    }

    #[test]
    fn rejection_converts_to_a_transient_core_error() {
        let e: Error = TriangulationRejection::LowParallax.into();
        assert!(e.is_transient());
        assert!(e.to_string().contains("parallax"));
    }
}
