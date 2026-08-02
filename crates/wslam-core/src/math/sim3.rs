//! Sim(3) — similarity transform, and Umeyama alignment.
//!
//! Needed for exactly one thing in the pipeline (spec.md §6, L3): "ATE after
//! Sim(3) alignment (scale-free — L3 does not claim scale)". Up-to-scale
//! trajectories can only be compared to metric ground truth after solving for
//! the similarity that best aligns them.

use super::{Mat3, Mat4, Scalar, Se3, So3, Vec3};

/// A similarity transform: rotation, translation, and a positive scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sim3 {
    rotation: So3,
    translation: Vec3,
    scale: Scalar,
}

impl Default for Sim3 {
    fn default() -> Self {
        Self::identity()
    }
}

impl Sim3 {
    /// The identity similarity.
    #[must_use]
    pub fn identity() -> Self {
        Sim3 {
            rotation: So3::identity(),
            translation: Vec3::zeros(),
            scale: 1.0,
        }
    }

    /// Build from parts. `scale` must be strictly positive.
    #[must_use]
    pub fn new(rotation: So3, translation: Vec3, scale: Scalar) -> Self {
        debug_assert!(scale > 0.0, "Sim3 scale must be positive, got {scale}");
        Sim3 {
            rotation,
            translation,
            scale,
        }
    }

    /// Rotation component.
    #[must_use]
    pub fn rotation(&self) -> So3 {
        self.rotation
    }
    /// Translation component.
    #[must_use]
    pub fn translation(&self) -> Vec3 {
        self.translation
    }
    /// Scale component.
    #[must_use]
    pub fn scale(&self) -> Scalar {
        self.scale
    }

    /// Apply: `s * R * p + t`.
    #[inline]
    #[must_use]
    pub fn act(&self, p: &Vec3) -> Vec3 {
        self.rotation.act(p) * self.scale + self.translation
    }

    /// Apply to a pose. Rotation composes; the camera centre is scaled.
    #[must_use]
    pub fn act_pose(&self, pose: &Se3) -> Se3 {
        Se3::new(
            self.rotation.compose(&pose.rotation()),
            self.act(&pose.translation()),
        )
    }

    /// Inverse similarity.
    #[must_use]
    pub fn inverse(&self) -> Self {
        let inv_s = 1.0 / self.scale;
        let r_inv = self.rotation.inverse();
        Sim3::new(r_inv, -(r_inv.act(&self.translation) * inv_s), inv_s)
    }

    /// As a 4x4 matrix with the scale folded into the rotation block.
    #[must_use]
    pub fn matrix(&self) -> Mat4 {
        let mut m = Mat4::identity();
        let sr: Mat3 = self.rotation.matrix() * self.scale;
        m.fixed_view_mut::<3, 3>(0, 0).copy_from(&sr);
        m[(0, 3)] = self.translation.x;
        m[(1, 3)] = self.translation.y;
        m[(2, 3)] = self.translation.z;
        m
    }
}

/// Outcome of [`umeyama`].
#[derive(Debug, Clone, Copy)]
pub struct Alignment {
    /// The similarity taking `source` points onto `target` points.
    pub transform: Sim3,
    /// Root-mean-square residual after alignment, in target units.
    pub rmse: Scalar,
}

/// Umeyama least-squares similarity alignment (Umeyama 1991).
///
/// Solves for `(s, R, t)` minimising `sum ||target_i - (s R source_i + t)||^2`.
/// With `estimate_scale = false` this degenerates to Horn's absolute
/// orientation, which is what a metric-to-metric ATE comparison wants.
///
/// Returns `None` if fewer than 3 correspondences are supplied, or if the
/// source point cloud is degenerate (all points coincident).
#[must_use]
pub fn umeyama(source: &[Vec3], target: &[Vec3], estimate_scale: bool) -> Option<Alignment> {
    let n = source.len();
    if n < 3 || target.len() != n {
        return None;
    }
    let inv_n = 1.0 / n as Scalar;

    let mu_src: Vec3 = source.iter().sum::<Vec3>() * inv_n;
    let mu_tgt: Vec3 = target.iter().sum::<Vec3>() * inv_n;

    let mut sigma_src = 0.0;
    let mut cov = Mat3::zeros();
    for (s, t) in source.iter().zip(target.iter()) {
        let ds = s - mu_src;
        let dt = t - mu_tgt;
        sigma_src += ds.norm_squared();
        cov += dt * ds.transpose();
    }
    sigma_src *= inv_n;
    cov *= inv_n;

    if sigma_src < 1e-18 {
        return None; // degenerate: source has no spatial extent
    }

    let svd = cov.svd(true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;
    let d = svd.singular_values;

    // Reflection guard: if det(U)*det(V) < 0 the naive solution is a reflection,
    // not a rotation. Flip the sign of the smallest singular direction.
    let mut s_mat = Mat3::identity();
    if u.determinant() * v_t.determinant() < 0.0 {
        s_mat[(2, 2)] = -1.0;
    }
    let rot = u * s_mat * v_t;

    let trace_ds = d[0] * s_mat[(0, 0)] + d[1] * s_mat[(1, 1)] + d[2] * s_mat[(2, 2)];
    let scale = if estimate_scale {
        trace_ds / sigma_src
    } else {
        1.0
    };
    if !(scale.is_finite() && scale > 0.0) {
        return None;
    }

    let rotation = So3::from_matrix(&rot);
    let translation = mu_tgt - rotation.act(&mu_src) * scale;
    let transform = Sim3::new(rotation, translation, scale);

    let mut sse = 0.0;
    for (s, t) in source.iter().zip(target.iter()) {
        sse += (t - transform.act(s)).norm_squared();
    }
    Some(Alignment {
        transform,
        rmse: (sse * inv_n).sqrt(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3;
    use approx::assert_relative_eq;

    #[test]
    fn recovers_known_similarity() {
        let truth = Sim3::new(
            So3::exp(&Vec3::new(0.2, -0.4, 0.1)),
            Vec3::new(3.0, -1.0, 0.5),
            2.75,
        );
        let src: Vec<Vec3> = (0..40)
            .map(|i| {
                let f = i as f64;
                Vec3::new(f.sin() * 2.0, (f * 0.7).cos(), f * 0.13)
            })
            .collect();
        let tgt: Vec<Vec3> = src.iter().map(|p| truth.act(p)).collect();

        let a = umeyama(&src, &tgt, true).expect("alignment");
        assert_relative_eq!(a.transform.scale(), 2.75, epsilon = 1e-9);
        assert_relative_eq!(
            a.transform.rotation().matrix(),
            truth.rotation().matrix(),
            epsilon = 1e-9
        );
        assert_relative_eq!(
            a.transform.translation(),
            truth.translation(),
            epsilon = 1e-8
        );
        assert!(a.rmse < 1e-9);
    }

    #[test]
    fn scale_locked_when_not_estimated() {
        let src = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let tgt: Vec<Vec3> = src.iter().map(|p| p * 3.0).collect();
        let a = umeyama(&src, &tgt, false).expect("alignment");
        assert_relative_eq!(a.transform.scale(), 1.0, epsilon = 1e-15);
        assert!(a.rmse > 0.5, "scale-locked fit should not fit a 3x scaling");
    }

    #[test]
    fn rejects_degenerate_input() {
        let p = vec![Vec3::zeros(); 5];
        assert!(umeyama(&p, &p, true).is_none());
        assert!(umeyama(&[Vec3::zeros()], &[Vec3::zeros()], true).is_none());
    }

    #[test]
    fn inverse_roundtrip() {
        let s = Sim3::new(
            So3::exp(&Vec3::new(0.1, 0.2, 0.3)),
            Vec3::new(1.0, 2.0, 3.0),
            1.7,
        );
        let p = Vec3::new(0.4, -0.9, 2.0);
        assert_relative_eq!(s.inverse().act(&s.act(&p)), p, epsilon = 1e-12);
    }
}
