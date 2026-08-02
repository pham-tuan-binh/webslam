//! SE(3) — rigid transform, `T_world_camera` by convention.

use super::{hat, join_twist, split_twist, Mat3, Mat4, Mat6, Scalar, So3, Vec3, Vec6, SMALL_ANGLE};

/// A rigid body transform in SE(3).
///
/// Interpreted throughout web-slam as `T_world_camera`: it takes a point in
/// camera coordinates to world coordinates, so [`Se3::translation`] is the
/// camera centre expressed in world coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Se3 {
    rotation: So3,
    translation: Vec3,
}

impl Default for Se3 {
    fn default() -> Self {
        Self::identity()
    }
}

impl Se3 {
    /// The identity transform.
    #[inline]
    #[must_use]
    pub fn identity() -> Self {
        Se3 {
            rotation: So3::identity(),
            translation: Vec3::zeros(),
        }
    }

    /// Build from a rotation and a translation.
    #[inline]
    #[must_use]
    pub fn new(rotation: So3, translation: Vec3) -> Self {
        Se3 {
            rotation,
            translation,
        }
    }

    /// Pure rotation.
    #[inline]
    #[must_use]
    pub fn from_rotation(rotation: So3) -> Self {
        Se3::new(rotation, Vec3::zeros())
    }

    /// Pure translation.
    #[inline]
    #[must_use]
    pub fn from_translation(translation: Vec3) -> Self {
        Se3::new(So3::identity(), translation)
    }

    /// Build from a 4x4 homogeneous matrix, projecting the rotation block to
    /// the nearest rotation.
    #[must_use]
    pub fn from_matrix(m: &Mat4) -> Self {
        let r = m.fixed_view::<3, 3>(0, 0).into_owned();
        let t = Vec3::new(m[(0, 3)], m[(1, 3)], m[(2, 3)]);
        Se3::new(So3::from_matrix(&r), t)
    }

    /// The rotation component.
    #[inline]
    #[must_use]
    pub fn rotation(&self) -> So3 {
        self.rotation
    }

    /// The translation component — the camera centre in world coordinates.
    #[inline]
    #[must_use]
    pub fn translation(&self) -> Vec3 {
        self.translation
    }

    /// Mutable access to the translation, for in-place scale application.
    #[inline]
    pub fn translation_mut(&mut self) -> &mut Vec3 {
        &mut self.translation
    }

    /// As a 4x4 homogeneous matrix (row/column indexed, not memory order).
    #[must_use]
    pub fn matrix(&self) -> Mat4 {
        let mut m = Mat4::identity();
        m.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&self.rotation.matrix());
        m[(0, 3)] = self.translation.x;
        m[(1, 3)] = self.translation.y;
        m[(2, 3)] = self.translation.z;
        m
    }

    /// Column-major `f32` 4x4, ready for `three.js` / WebGPU without transposing.
    ///
    /// `nalgebra` is already column-major in memory, so this is a straight cast
    /// of the storage order — but the test suite pins it, because getting this
    /// wrong produces a plausible-looking transposed camera that is very hard to
    /// spot by eye.
    #[must_use]
    pub fn to_matrix_f32(&self) -> [f32; 16] {
        let m = self.matrix();
        let mut out = [0f32; 16];
        for col in 0..4 {
            for row in 0..4 {
                out[col * 4 + row] = m[(row, col)] as f32;
            }
        }
        out
    }

    /// Inverse transform.
    #[must_use]
    pub fn inverse(&self) -> Self {
        let r_inv = self.rotation.inverse();
        Se3::new(r_inv, -(r_inv.act(&self.translation)))
    }

    /// Group composition, `self * other`.
    #[must_use]
    pub fn compose(&self, other: &Self) -> Self {
        Se3::new(
            self.rotation.compose(&other.rotation),
            self.rotation.act(&other.translation) + self.translation,
        )
    }

    /// Apply to a point.
    #[inline]
    #[must_use]
    pub fn act(&self, p: &Vec3) -> Vec3 {
        self.rotation.act(p) + self.translation
    }

    /// Exponential map from `se(3)`. Twist is `[rho; phi]`.
    #[must_use]
    pub fn exp(xi: &Vec6) -> Self {
        let (rho, phi) = split_twist(xi);
        let rotation = So3::exp(&phi);
        let v = Self::left_jacobian_so3(&phi);
        Se3::new(rotation, v * rho)
    }

    /// Logarithm map to `se(3)`. Returns `[rho; phi]`.
    #[must_use]
    pub fn log(&self) -> Vec6 {
        let phi = self.rotation.log();
        let v_inv = Self::left_jacobian_so3_inv(&phi);
        join_twist(&(v_inv * self.translation), &phi)
    }

    /// The `V` matrix of the SE(3) exponential — the SO(3) left Jacobian.
    fn left_jacobian_so3(phi: &Vec3) -> Mat3 {
        let theta_sq = phi.norm_squared();
        let w = hat(phi);
        if theta_sq < SMALL_ANGLE * SMALL_ANGLE {
            Mat3::identity() + 0.5 * w + (1.0 / 6.0) * w * w
        } else {
            let theta = theta_sq.sqrt();
            let a = (1.0 - theta.cos()) / theta_sq;
            let b = (theta - theta.sin()) / (theta_sq * theta);
            Mat3::identity() + a * w + b * w * w
        }
    }

    fn left_jacobian_so3_inv(phi: &Vec3) -> Mat3 {
        let theta_sq = phi.norm_squared();
        let w = hat(phi);
        if theta_sq < SMALL_ANGLE * SMALL_ANGLE {
            Mat3::identity() - 0.5 * w + (1.0 / 12.0) * w * w
        } else {
            let theta = theta_sq.sqrt();
            let c = 1.0 / theta_sq - (1.0 + theta.cos()) / (2.0 * theta * theta.sin());
            Mat3::identity() - 0.5 * w + c * w * w
        }
    }

    /// Adjoint, in `[rho; phi]` block order:
    ///
    /// ```text
    /// Adj = [ R   hat(t) R ]
    ///       [ 0       R    ]
    /// ```
    #[must_use]
    pub fn adjoint(&self) -> Mat6 {
        let r = self.rotation.matrix();
        let tr = hat(&self.translation) * r;
        let mut m = Mat6::zeros();
        m.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
        m.fixed_view_mut::<3, 3>(0, 3).copy_from(&tr);
        m.fixed_view_mut::<3, 3>(3, 3).copy_from(&r);
        m
    }

    /// Right-plus: `self * exp(delta)`.
    #[inline]
    #[must_use]
    pub fn plus(&self, delta: &Vec6) -> Self {
        self.compose(&Self::exp(delta))
    }

    /// Right-minus: the `delta` such that `other.plus(delta) == self`.
    #[inline]
    #[must_use]
    pub fn minus(&self, other: &Self) -> Vec6 {
        other.inverse().compose(self).log()
    }

    /// Scale the translation component only — what a [`crate::ScaleKind`]
    /// applies to an up-to-scale trajectory. Rotation is scale-invariant.
    #[must_use]
    pub fn scaled(&self, s: Scalar) -> Self {
        Se3::new(self.rotation, self.translation * s)
    }

    /// Interpolate between two poses: slerp on rotation, lerp on translation.
    /// `t` is clamped to `[0, 1]`.
    #[must_use]
    pub fn interpolate(&self, other: &Self, t: Scalar) -> Self {
        let t = t.clamp(0.0, 1.0);
        Se3::new(
            self.rotation.slerp(&other.rotation, t),
            self.translation * (1.0 - t) + other.translation * t,
        )
    }

    /// Renormalise the rotation. Cheap; call after long integration runs.
    #[must_use]
    pub fn normalized(&self) -> Self {
        Se3::new(self.rotation.normalized(), self.translation)
    }
}

impl std::ops::Mul for Se3 {
    type Output = Se3;
    #[inline]
    fn mul(self, rhs: Se3) -> Se3 {
        self.compose(&rhs)
    }
}

impl std::ops::Mul<Vec3> for Se3 {
    type Output = Vec3;
    #[inline]
    fn mul(self, rhs: Vec3) -> Vec3 {
        self.act(&rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn sample() -> Se3 {
        Se3::new(
            So3::exp(&Vec3::new(0.3, -0.2, 0.7)),
            Vec3::new(1.5, -0.4, 2.2),
        )
    }

    #[test]
    fn exp_log_roundtrip() {
        for xi in [
            Vec6::new(0.1, 0.2, 0.3, 0.05, -0.15, 0.25),
            Vec6::new(-2.0, 5.0, 0.5, 1.2, -0.4, 0.9),
            Vec6::new(1e-10, -1e-10, 0.0, 1e-11, 0.0, 0.0),
            Vec6::zeros(),
        ] {
            let t = Se3::exp(&xi);
            assert_relative_eq!(t.log(), xi, epsilon = 1e-11);
        }
    }

    #[test]
    fn log_exp_roundtrip() {
        let t = sample();
        let back = Se3::exp(&t.log());
        assert_relative_eq!(back.matrix(), t.matrix(), epsilon = 1e-12);
    }

    #[test]
    fn inverse_is_inverse() {
        let t = sample();
        assert_relative_eq!(
            t.compose(&t.inverse()).matrix(),
            Mat4::identity(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn compose_matches_matrix_product() {
        let a = sample();
        let b = Se3::new(
            So3::exp(&Vec3::new(-0.5, 0.1, 0.2)),
            Vec3::new(-1.0, 3.0, 0.5),
        );
        assert_relative_eq!(
            a.compose(&b).matrix(),
            a.matrix() * b.matrix(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn adjoint_identity() {
        // Adj(T) * xi == log(T * exp(xi) * T^-1)
        let t = sample();
        let xi = Vec6::new(0.02, -0.01, 0.03, 0.004, 0.005, -0.006);
        let lhs = t.adjoint() * xi;
        let rhs = t.compose(&Se3::exp(&xi)).compose(&t.inverse()).log();
        assert_relative_eq!(lhs, rhs, epsilon = 1e-9);
    }

    #[test]
    fn column_major_export_is_not_transposed() {
        // Element [row=0, col=3] of a column-major 4x4 is index 12: the x
        // translation. A transposed export would put it at index 3.
        let t = Se3::from_translation(Vec3::new(7.0, 8.0, 9.0));
        let m = t.to_matrix_f32();
        assert_eq!(m[12], 7.0);
        assert_eq!(m[13], 8.0);
        assert_eq!(m[14], 9.0);
        assert_eq!(m[3], 0.0);
    }

    #[test]
    fn plus_minus_are_inverse() {
        let a = sample();
        let d = Vec6::new(0.01, 0.02, -0.01, 0.003, -0.002, 0.001);
        assert_relative_eq!(a.plus(&d).minus(&a), d, epsilon = 1e-12);
    }

    #[test]
    fn scaled_leaves_rotation_alone() {
        let t = sample();
        let s = t.scaled(2.5);
        assert_relative_eq!(
            s.rotation().matrix(),
            t.rotation().matrix(),
            epsilon = 1e-15
        );
        assert_relative_eq!(s.translation(), t.translation() * 2.5, epsilon = 1e-15);
    }
}
