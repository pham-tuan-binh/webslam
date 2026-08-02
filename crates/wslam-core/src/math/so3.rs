//! SO(3) — 3D rotation, stored as a unit quaternion.

use super::{hat, Mat3, Quat, Scalar, Vec3, SMALL_ANGLE};

/// A rotation in SO(3).
///
/// Stored as a unit quaternion because renormalisation is one cheap division
/// and the exp/log maps are numerically well behaved near identity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct So3(Quat);

impl Default for So3 {
    fn default() -> Self {
        Self::identity()
    }
}

impl So3 {
    /// The identity rotation.
    #[inline]
    #[must_use]
    pub fn identity() -> Self {
        So3(Quat::identity())
    }

    /// Wrap an already-normalised quaternion.
    #[inline]
    #[must_use]
    pub fn from_quaternion(q: Quat) -> Self {
        So3(q)
    }

    /// Build from raw `(w, x, y, z)`, normalising.
    #[must_use]
    pub fn from_wxyz(w: Scalar, x: Scalar, y: Scalar, z: Scalar) -> Self {
        So3(Quat::from_quaternion(nalgebra::Quaternion::new(w, x, y, z)))
    }

    /// Build from a rotation matrix, projecting to the nearest rotation.
    #[must_use]
    pub fn from_matrix(m: &Mat3) -> Self {
        So3(Quat::from_rotation_matrix(
            &nalgebra::Rotation3::from_matrix_eps(m, 1e-12, 100, nalgebra::Rotation3::identity()),
        ))
    }

    /// The underlying unit quaternion.
    #[inline]
    #[must_use]
    pub fn quaternion(&self) -> Quat {
        self.0
    }

    /// As a 3x3 rotation matrix.
    #[inline]
    #[must_use]
    pub fn matrix(&self) -> Mat3 {
        self.0.to_rotation_matrix().into_inner()
    }

    /// Inverse rotation.
    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Self {
        So3(self.0.inverse())
    }

    /// Group composition, `self * other`.
    #[inline]
    #[must_use]
    pub fn compose(&self, other: &Self) -> Self {
        So3(self.0 * other.0)
    }

    /// Rotate a vector.
    #[inline]
    #[must_use]
    pub fn act(&self, v: &Vec3) -> Vec3 {
        self.0 * v
    }

    /// Exponential map from `so(3)` (a rotation vector, axis * angle in radians).
    ///
    /// Uses a Taylor expansion below [`SMALL_ANGLE`]; the closed form divides by
    /// `theta` and loses all precision at the origin.
    #[must_use]
    pub fn exp(phi: &Vec3) -> Self {
        let theta_sq = phi.norm_squared();
        if theta_sq < SMALL_ANGLE * SMALL_ANGLE {
            // sin(t/2)/t -> 1/2 - t^2/48, cos(t/2) -> 1 - t^2/8
            let half = 0.5 - theta_sq / 48.0;
            let w = 1.0 - theta_sq / 8.0;
            So3::from_wxyz(w, phi.x * half, phi.y * half, phi.z * half)
        } else {
            let theta = theta_sq.sqrt();
            let half = theta * 0.5;
            let s = half.sin() / theta;
            So3::from_wxyz(half.cos(), phi.x * s, phi.y * s, phi.z * s)
        }
    }

    /// Logarithm map to `so(3)`. Returns a rotation vector with `|phi| <= pi`.
    #[must_use]
    pub fn log(&self) -> Vec3 {
        // Canonicalise to the hemisphere w >= 0 so the result is the shortest
        // rotation; q and -q are the same rotation but log differs by 2*pi.
        let q = if self.0.w < 0.0 {
            nalgebra::Quaternion::new(-self.0.w, -self.0.i, -self.0.j, -self.0.k)
        } else {
            *self.0.as_ref()
        };
        let v = Vec3::new(q.i, q.j, q.k);
        let n = v.norm();
        if n < SMALL_ANGLE {
            // 2*atan(n/w)/n -> 2/w * (1 - n^2/(3 w^2))
            let two_over_w = 2.0 / q.w;
            v * (two_over_w - two_over_w * n * n / (3.0 * q.w * q.w))
        } else {
            v * (2.0 * n.atan2(q.w) / n)
        }
    }

    /// Adjoint. For SO(3) this is just the rotation matrix.
    #[inline]
    #[must_use]
    pub fn adjoint(&self) -> Mat3 {
        self.matrix()
    }

    /// Right Jacobian `Jr(phi)`, satisfying
    /// `exp(phi + dphi) ~= exp(phi) * exp(Jr(phi) * dphi)`.
    #[must_use]
    pub fn right_jacobian(phi: &Vec3) -> Mat3 {
        let theta_sq = phi.norm_squared();
        let w = hat(phi);
        if theta_sq < SMALL_ANGLE * SMALL_ANGLE {
            Mat3::identity() - 0.5 * w + (1.0 / 6.0) * w * w
        } else {
            let theta = theta_sq.sqrt();
            let a = (1.0 - theta.cos()) / theta_sq;
            let b = (theta - theta.sin()) / (theta_sq * theta);
            Mat3::identity() - a * w + b * w * w
        }
    }

    /// Inverse of the right Jacobian.
    #[must_use]
    pub fn right_jacobian_inv(phi: &Vec3) -> Mat3 {
        let theta_sq = phi.norm_squared();
        let w = hat(phi);
        if theta_sq < SMALL_ANGLE * SMALL_ANGLE {
            Mat3::identity() + 0.5 * w + (1.0 / 12.0) * w * w
        } else {
            let theta = theta_sq.sqrt();
            let half = 0.5 * theta;
            // (1/theta^2) - (1 + cos t) / (2 t sin t)
            let c = 1.0 / theta_sq - (1.0 + theta.cos()) / (2.0 * theta * theta.sin());
            let _ = half;
            Mat3::identity() + 0.5 * w + c * w * w
        }
    }

    /// Left Jacobian `Jl(phi) = Jr(-phi)`.
    #[inline]
    #[must_use]
    pub fn left_jacobian(phi: &Vec3) -> Mat3 {
        Self::right_jacobian(&(-phi))
    }

    /// Inverse of the left Jacobian.
    #[inline]
    #[must_use]
    pub fn left_jacobian_inv(phi: &Vec3) -> Mat3 {
        Self::right_jacobian_inv(&(-phi))
    }

    /// Right-plus: `self * exp(delta)`.
    #[inline]
    #[must_use]
    pub fn plus(&self, delta: &Vec3) -> Self {
        self.compose(&Self::exp(delta))
    }

    /// Right-minus: the `delta` such that `other.plus(delta) == self`.
    #[inline]
    #[must_use]
    pub fn minus(&self, other: &Self) -> Vec3 {
        other.inverse().compose(self).log()
    }

    /// Rotation angle in radians, in `[0, pi]`.
    #[inline]
    #[must_use]
    pub fn angle(&self) -> Scalar {
        self.log().norm()
    }

    /// Renormalise the quaternion. Cheap; call it after long integration runs.
    #[must_use]
    pub fn normalized(&self) -> Self {
        So3(Quat::from_quaternion(*self.0.as_ref()))
    }

    /// Spherical linear interpolation, `t` in `[0, 1]`.
    #[must_use]
    pub fn slerp(&self, other: &Self, t: Scalar) -> Self {
        So3(self.0.slerp(&other.0, t))
    }

    /// The gravity-aligned rotation that maps the measured accelerometer
    /// direction onto the world `-Z`... deliberately *not* provided here:
    /// gravity convention belongs to L1, which owns the frame definition.
    ///
    /// Builds the shortest rotation taking `from` onto `to` (both are
    /// normalised internally). Returns identity if either is degenerate.
    #[must_use]
    pub fn rotation_between(from: &Vec3, to: &Vec3) -> Self {
        match Quat::rotation_between(from, to) {
            Some(q) => So3(q),
            None => So3::identity(),
        }
    }
}

impl std::ops::Mul for So3 {
    type Output = So3;
    #[inline]
    fn mul(self, rhs: So3) -> So3 {
        self.compose(&rhs)
    }
}

impl std::ops::Mul<Vec3> for So3 {
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

    #[test]
    fn exp_log_roundtrip_generic() {
        for phi in [
            Vec3::new(0.1, -0.2, 0.35),
            Vec3::new(1.0, 2.0, -0.5),
            Vec3::new(0.0, 0.0, std::f64::consts::PI * 0.99),
        ] {
            let r = So3::exp(&phi);
            assert_relative_eq!(r.log(), phi, epsilon = 1e-12);
        }
    }

    #[test]
    fn exp_log_roundtrip_near_identity() {
        // The small-angle branch is where hand-rolled Lie code usually dies.
        for scale in [1e-10, 1e-9, 1e-8, 1e-7, 1e-6] {
            let phi = Vec3::new(1.0, -2.0, 0.5).normalize() * scale;
            let r = So3::exp(&phi);
            assert_relative_eq!(r.log(), phi, epsilon = 1e-18, max_relative = 1e-9);
        }
    }

    #[test]
    fn exp_matches_rodrigues() {
        let phi = Vec3::new(0.3, -0.7, 0.2);
        let theta = phi.norm();
        let k = phi / theta;
        let kx = hat(&k);
        let rodrigues = Mat3::identity() + theta.sin() * kx + (1.0 - theta.cos()) * kx * kx;
        assert_relative_eq!(So3::exp(&phi).matrix(), rodrigues, epsilon = 1e-12);
    }

    #[test]
    fn right_jacobian_matches_numerical() {
        let phi = Vec3::new(0.4, 0.1, -0.6);
        let jr = So3::right_jacobian(&phi);
        let eps = 1e-6;
        for i in 0..3 {
            let mut d = Vec3::zeros();
            d[i] = eps;
            // exp(phi + d) = exp(phi) * exp(Jr * d)
            let lhs = So3::exp(&(phi + d));
            let rhs = So3::exp(&phi).compose(&So3::exp(&(jr * d)));
            assert_relative_eq!(lhs.log(), rhs.log(), epsilon = 1e-9);
        }
    }

    #[test]
    fn right_jacobian_inv_is_inverse() {
        for phi in [
            Vec3::new(0.4, 0.1, -0.6),
            Vec3::new(1e-10, 0.0, 0.0),
            Vec3::new(2.0, -1.0, 0.5),
        ] {
            let p = So3::right_jacobian(&phi) * So3::right_jacobian_inv(&phi);
            assert_relative_eq!(p, Mat3::identity(), epsilon = 1e-9);
        }
    }

    #[test]
    fn log_takes_shortest_path() {
        // q and -q are the same rotation; log must return |phi| <= pi.
        let phi = Vec3::new(0.0, 0.0, 3.0);
        let r = So3::exp(&phi);
        let neg = So3::from_wxyz(
            -r.quaternion().w,
            -r.quaternion().i,
            -r.quaternion().j,
            -r.quaternion().k,
        );
        assert_relative_eq!(neg.log(), phi, epsilon = 1e-12);
        assert!(r.log().norm() <= std::f64::consts::PI + 1e-12);
    }

    #[test]
    fn plus_minus_are_inverse() {
        let a = So3::exp(&Vec3::new(0.2, 0.3, -0.1));
        let d = Vec3::new(0.01, -0.02, 0.03);
        let b = a.plus(&d);
        assert_relative_eq!(b.minus(&a), d, epsilon = 1e-12);
    }
}
