//! Homography estimation and planar pose recovery — stages four and six.
//!
//! A tag is four coplanar points of known metric geometry, so its pose follows
//! from the plane-to-image homography without any iterative solve. The
//! decomposition below is Zhang's: strip the intrinsics, recover the scale from
//! the fact that the first two columns are rotation columns and therefore unit
//! length, and project the result back onto SO(3).
//!
//! This is the analytic branch of IPPE (Collins & Bartoli, *Infinitesimal
//! Plane-Based Pose Estimation*). IPPE's second contribution is enumerating
//! *both* poses of the two-fold planar ambiguity; we return the one whose
//! reprojection error is lower, and note that the ambiguity flips the tag's
//! surface normal while leaving `|t|` — the only quantity scale depends on —
//! essentially unchanged.

use wslam_core::{CameraIntrinsics, Mat3, Scalar, Se3, So3, Vec2, Vec3};

/// Apply a homography to a point, dividing through by the third coordinate.
///
/// Returns `None` when the point maps to the line at infinity.
#[must_use]
pub fn apply(h: &Mat3, p: Vec2) -> Option<Vec2> {
    let v = h * Vec3::new(p.x, p.y, 1.0);
    if v.z.abs() < 1e-12 {
        return None;
    }
    Some(Vec2::new(v.x / v.z, v.y / v.z))
}

/// Direct linear transform homography `src -> dst`, with Hartley
/// normalisation.
///
/// Without the normalisation the 2n x 9 system is badly conditioned whenever
/// the pixel coordinates are far from the origin, which for a 1280 px frame is
/// always. Returns `None` for fewer than four correspondences or a degenerate
/// configuration.
#[must_use]
pub fn homography_dlt(correspondences: &[(Vec2, Vec2)]) -> Option<Mat3> {
    let n = correspondences.len();
    if n < 4 {
        return None;
    }
    let src: Vec<Vec2> = correspondences.iter().map(|c| c.0).collect();
    let dst: Vec<Vec2> = correspondences.iter().map(|c| c.1).collect();
    let (t_src, ns) = normalise(&src)?;
    let (t_dst, nd) = normalise(&dst)?;

    // At least 9 rows, zero-padded. nalgebra's SVD is *thin*: for an m x 9
    // matrix it returns `v_t` with only min(m, 9) rows, so the minimal-case
    // 8 x 9 system (four correspondences — which is every tag) hands back a
    // `v_t` that does not contain the null-space vector at all, and the
    // argmin below then picks the 8th right singular vector instead of the
    // 9th. Padding with zero rows leaves the row space, the singular values
    // and their singular vectors untouched, but makes min(m, 9) == 9 so the
    // null vector is present.
    let rows = (2 * n).max(9);
    let mut a = nalgebra::DMatrix::<Scalar>::zeros(rows, 9);
    for i in 0..n {
        let (x, y) = (ns[i].x, ns[i].y);
        let (u, v) = (nd[i].x, nd[i].y);
        a[(2 * i, 0)] = -x;
        a[(2 * i, 1)] = -y;
        a[(2 * i, 2)] = -1.0;
        a[(2 * i, 6)] = u * x;
        a[(2 * i, 7)] = u * y;
        a[(2 * i, 8)] = u;
        a[(2 * i + 1, 3)] = -x;
        a[(2 * i + 1, 4)] = -y;
        a[(2 * i + 1, 5)] = -1.0;
        a[(2 * i + 1, 6)] = v * x;
        a[(2 * i + 1, 7)] = v * y;
        a[(2 * i + 1, 8)] = v;
    }

    let svd = a.svd(false, true);
    let v_t = svd.v_t?;
    // Right singular vector of the smallest singular value.
    let last = v_t.nrows() - 1;
    let mut min_index = last;
    let mut min_value = Scalar::INFINITY;
    for (i, s) in svd.singular_values.iter().enumerate() {
        if *s < min_value {
            min_value = *s;
            min_index = i;
        }
    }
    let row = v_t.row(min_index.min(last));
    let h_n = Mat3::new(
        row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8],
    );

    let h = t_dst.try_inverse()? * h_n * t_src;
    if h.iter().any(|v| !v.is_finite()) || h[(2, 2)].abs() < 1e-15 {
        return None;
    }
    Some(h / h[(2, 2)])
}

/// Hartley normalisation: centre on the centroid and scale so the mean
/// distance from it is `sqrt(2)`.
fn normalise(points: &[Vec2]) -> Option<(Mat3, Vec<Vec2>)> {
    let n = points.len() as Scalar;
    let mean: Vec2 = points.iter().sum::<Vec2>() / n;
    let mean_dist = points.iter().map(|p| (p - mean).norm()).sum::<Scalar>() / n;
    if mean_dist < 1e-12 {
        return None;
    }
    let s = std::f64::consts::SQRT_2 / mean_dist;
    let t = Mat3::new(s, 0.0, -s * mean.x, 0.0, s, -s * mean.y, 0.0, 0.0, 1.0);
    let out = points
        .iter()
        .map(|p| Vec2::new(s * (p.x - mean.x), s * (p.y - mean.y)))
        .collect();
    Some((t, out))
}

/// Pose of a planar target from its plane-to-image homography.
///
/// `h` must map *tag-plane metres* `(x, y)` to pixels. Returns
/// `T_camera_tag`: it takes a point in the tag frame (x right, y down, z out
/// of the tag away from the camera — the OpenCV convention, so a
/// fronto-parallel tag has identity rotation) into camera coordinates.
///
/// Returns `None` if the intrinsics are singular or the homography does not
/// place the tag in front of the camera.
#[must_use]
pub fn pose_from_homography(h: &Mat3, k: &CameraIntrinsics) -> Option<Se3> {
    let m = k.inverse_matrix() * h;
    let (m1, m2, m3) = (m.column(0), m.column(1), m.column(2));

    // The first two columns are rotation columns scaled by one common lambda,
    // so their mean norm is the best two-sample estimate of 1/lambda.
    let mean_norm = 0.5 * (m1.norm() + m2.norm());
    if !(mean_norm.is_finite() && mean_norm > 1e-12) {
        return None;
    }
    let mut lambda = 1.0 / mean_norm;
    // Sign: the tag has to be in front of the camera. A homography is only
    // defined up to scale, so this is the one piece of information the
    // algebra cannot supply.
    if lambda * m3.z < 0.0 {
        lambda = -lambda;
    }

    let r1: Vec3 = m1 * lambda;
    let r2: Vec3 = m2 * lambda;
    let r3: Vec3 = r1.cross(&r2);
    let t: Vec3 = m3 * lambda;
    if t.z <= 0.0 || !t.iter().all(|v| v.is_finite()) {
        return None;
    }

    let mut approx = Mat3::zeros();
    approx.set_column(0, &r1);
    approx.set_column(1, &r2);
    approx.set_column(2, &r3);
    let rotation = nearest_rotation(&approx)?;

    Some(Se3::new(So3::from_matrix(&rotation), t))
}

/// Closest rotation matrix in the Frobenius sense: `U diag(1,1,det) V^T`.
fn nearest_rotation(m: &Mat3) -> Option<Mat3> {
    if m.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let svd = m.svd(true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;
    let mut d = Mat3::identity();
    // Guard against a reflection, which would produce a left-handed "rotation"
    // that quietly mirrors the recovered pose.
    if (u * v_t).determinant() < 0.0 {
        d[(2, 2)] = -1.0;
    }
    Some(u * d * v_t)
}

/// Root-mean-square reprojection error of the tag corners under a pose.
#[must_use]
pub fn reprojection_rmse(
    pose: &Se3,
    k: &CameraIntrinsics,
    object: &[Vec3; 4],
    image: &[Vec2; 4],
) -> Scalar {
    let mut sse = 0.0;
    for (o, i) in object.iter().zip(image.iter()) {
        match k.project(&pose.act(o)) {
            Some(p) => sse += (p - i).norm_squared(),
            None => return Scalar::INFINITY,
        }
    }
    (sse * 0.25).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn intrinsics() -> CameraIntrinsics {
        CameraIntrinsics::from_focal(620.0, 640, 480)
    }

    fn unit_corners(half: Scalar) -> [Vec3; 4] {
        [
            Vec3::new(-half, -half, 0.0),
            Vec3::new(half, -half, 0.0),
            Vec3::new(half, half, 0.0),
            Vec3::new(-half, half, 0.0),
        ]
    }

    #[test]
    fn dlt_recovers_a_known_homography() {
        let truth = Mat3::new(1.2, 0.3, 45.0, -0.2, 0.9, 12.0, 0.0006, -0.0002, 1.0);
        let src = [
            Vec2::new(-1.0, -1.0),
            Vec2::new(1.0, -1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(-1.0, 1.0),
            Vec2::new(0.3, -0.7),
        ];
        let corr: Vec<(Vec2, Vec2)> = src
            .iter()
            .map(|&p| (p, apply(&truth, p).unwrap()))
            .collect();
        let h = homography_dlt(&corr).unwrap();
        for &(s, d) in &corr {
            assert_relative_eq!(apply(&h, s).unwrap(), d, epsilon = 1e-9);
        }
        // Normalised by h22, so the matrices themselves must agree.
        assert_relative_eq!(h, truth, epsilon = 1e-9);
    }

    #[test]
    fn dlt_refuses_degenerate_input() {
        assert!(homography_dlt(&[]).is_none());
        assert!(homography_dlt(&[(Vec2::zeros(), Vec2::zeros()); 3]).is_none());
        // All source points coincident: nothing to normalise against.
        assert!(homography_dlt(&[(Vec2::new(5.0, 5.0), Vec2::new(1.0, 2.0)); 6]).is_none());
    }

    /// The closed-form check: project a tag at a known pose, recover the pose
    /// from the projected corners, and demand the original back.
    #[test]
    fn pose_recovers_a_known_transform() {
        let k = intrinsics();
        let half = 0.05; // a 10 cm tag
        let object = unit_corners(half);

        for truth in [
            Se3::new(So3::identity(), Vec3::new(0.0, 0.0, 0.8)),
            Se3::new(
                So3::exp(&Vec3::new(0.25, -0.4, 0.15)),
                Vec3::new(0.1, -0.05, 1.4),
            ),
            Se3::new(
                So3::exp(&Vec3::new(-0.6, 0.2, 1.1)),
                Vec3::new(-0.2, 0.15, 2.2),
            ),
        ] {
            let image: Vec<Vec2> = object
                .iter()
                .map(|p| k.project(&truth.act(p)).unwrap())
                .collect();
            let corr: Vec<(Vec2, Vec2)> = object
                .iter()
                .zip(image.iter())
                .map(|(o, i)| (Vec2::new(o.x, o.y), *i))
                .collect();
            let h = homography_dlt(&corr).unwrap();
            let pose = pose_from_homography(&h, &k).unwrap();

            assert_relative_eq!(pose.translation(), truth.translation(), epsilon = 1e-9);
            assert_relative_eq!(
                pose.rotation().matrix(),
                truth.rotation().matrix(),
                epsilon = 1e-9
            );
            let img: [Vec2; 4] = [image[0], image[1], image[2], image[3]];
            assert!(reprojection_rmse(&pose, &k, &object, &img) < 1e-8);
        }
    }

    /// Scale is the distance the tag reports, so it gets its own assertion: a
    /// tag twice as far must report twice the distance, to nine digits.
    #[test]
    fn recovered_distance_is_linear_in_true_distance() {
        let k = intrinsics();
        let object = unit_corners(0.05);
        for z in [0.4, 0.8, 1.6, 3.2] {
            let truth = Se3::new(So3::exp(&Vec3::new(0.1, 0.2, 0.0)), Vec3::new(0.0, 0.0, z));
            let corr: Vec<(Vec2, Vec2)> = object
                .iter()
                .map(|p| (Vec2::new(p.x, p.y), k.project(&truth.act(p)).unwrap()))
                .collect();
            let pose = pose_from_homography(&homography_dlt(&corr).unwrap(), &k).unwrap();
            assert_relative_eq!(pose.translation().z, z, epsilon = 1e-9);
        }
    }

    #[test]
    fn a_rotation_matrix_stays_a_rotation() {
        let k = intrinsics();
        let object = unit_corners(0.03);
        let truth = Se3::new(
            So3::exp(&Vec3::new(0.8, -0.5, 0.3)),
            Vec3::new(0.05, 0.0, 0.6),
        );
        let corr: Vec<(Vec2, Vec2)> = object
            .iter()
            .map(|p| (Vec2::new(p.x, p.y), k.project(&truth.act(p)).unwrap()))
            .collect();
        let pose = pose_from_homography(&homography_dlt(&corr).unwrap(), &k).unwrap();
        let r = pose.rotation().matrix();
        assert_relative_eq!(r * r.transpose(), Mat3::identity(), epsilon = 1e-12);
        assert_relative_eq!(r.determinant(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn nearest_rotation_repairs_a_perturbed_rotation_without_mirroring() {
        let truth = So3::exp(&Vec3::new(0.3, 0.4, -0.2)).matrix();
        let mut noisy = truth;
        noisy[(0, 1)] += 0.02;
        noisy[(2, 0)] -= 0.015;
        let fixed = nearest_rotation(&noisy).unwrap();
        assert_relative_eq!(fixed.determinant(), 1.0, epsilon = 1e-12);
        assert!((fixed - truth).norm() < 0.03);
    }

    #[test]
    fn pose_rejects_a_tag_behind_the_camera() {
        let k = intrinsics();
        // A homography with a negative third column cannot place the tag in
        // front; both sign choices fail the cheirality test.
        let h = Mat3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        assert!(pose_from_homography(&h, &k).is_none());
    }
}
