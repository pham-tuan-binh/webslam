//! World-frame convention, and the tilt/yaw decomposition the filter is built on.
//!
//! `wslam_core::math::So3` deliberately refuses to define gravity ("gravity
//! convention belongs to L1, which owns the frame definition"), so this module
//! is where it is fixed, once:
//!
//! - The world frame is **Z-up**. Gravitational acceleration points along `-Z`.
//! - The accelerometer reports **specific force**, so a device at rest measures
//!   `+g` along world `+Z`. That is the convention `wslam_core` already tests
//!   against (`crates/wslam-core/src/window.rs` builds static samples as
//!   `accel = (0, 0, GRAVITY)`).
//! - Attitude is `R_world_body`, and errors are right-multiplicative,
//!   `R_true = R_est · exp(δθ)`, matching the workspace convention in
//!   `wslam_core::math` (`T ⊞ δ = T · exp(δ)`).
//!
//! The one non-obvious consequence is [`yaw_axis_body`]: a *world* yaw increment
//! `dψ` about `+Z` is the *body* right-perturbation `dψ · (R^T e_z)`, because
//! `Rz(dψ) R = R exp(dψ R^T e_z)`. That vector is the direction gravity cannot
//! observe, and the filter uses it in exactly two places — the null space of the
//! accelerometer Jacobian, and the row space of the yaw correction.

use wslam_core::math::{Scalar, So3, Vec3};

/// World up, `+Z`.
#[inline]
#[must_use]
pub fn world_up() -> Vec3 {
    Vec3::z()
}

/// Specific force measured by a stationary device, in world coordinates.
#[inline]
#[must_use]
pub fn world_specific_force() -> Vec3 {
    world_up() * wslam_core::imu::GRAVITY
}

/// World up expressed in body coordinates, `R^T e_z`.
///
/// This is both the direction the accelerometer predicts (as a unit vector) and
/// the unobservable direction of the attitude error. Those being the same vector
/// is not a coincidence: it is why gravity constrains exactly two of three
/// rotational degrees of freedom.
#[inline]
#[must_use]
pub fn yaw_axis_body(attitude: &So3) -> Vec3 {
    attitude.inverse().act(&world_up())
}

/// Unit vector along gravitational acceleration, in body coordinates.
///
/// Points **down**, so it is the negation of the normalised static
/// accelerometer reading. A level device with body axes aligned to world gives
/// `(0, 0, -1)`.
#[inline]
#[must_use]
pub fn gravity_direction_body(attitude: &So3) -> Vec3 {
    -yaw_axis_body(attitude)
}

/// Shortest rotation taking `from` onto `to`; both are normalised internally.
///
/// `So3::rotation_between` collapses to identity when `nalgebra` cannot form the
/// axis, which silently swallows the antiparallel case. A phone rolled through
/// 180 degrees is an ordinary orientation, and initialising it as level would be
/// a 180-degree error, so the antiparallel branch is handled explicitly here.
/// No choice of axis is more correct than another there, so the choice is made
/// deterministically rather than left to floating-point luck.
#[must_use]
pub fn rotation_aligning(from: &Vec3, to: &Vec3) -> So3 {
    let (a, b) = match (unit_or_none(from), unit_or_none(to)) {
        (Some(a), Some(b)) => (a, b),
        _ => return So3::identity(),
    };
    let cos = a.dot(&b).clamp(-1.0, 1.0);
    if cos > 1.0 - 1e-15 {
        return So3::identity();
    }
    if cos < -1.0 + 1e-12 {
        return So3::exp(&(any_orthogonal(&a) * std::f64::consts::PI));
    }
    let axis = a.cross(&b);
    // atan2 of |sin| against cos, rather than acos, so the angle stays accurate
    // near 0 and near pi where acos loses half its digits.
    let angle = axis.norm().atan2(cos);
    So3::exp(&(axis.normalize() * angle))
}

/// World yaw of an attitude, in radians, from the tilt-twist decomposition
/// `R = Rz(ψ) · T` where `T` is the minimal (yaw-free) tilt.
///
/// Not the ZYX-Euler yaw: that one is `atan2(R[(1,0)], R[(0,0)])`, which is a
/// different function of `R` as soon as the device is tilted, and degenerates at
/// 90 degrees of pitch — which a phone held to look at the floor reaches
/// routinely. This definition is the one that pairs with [`yaw_axis_body`]: it
/// is invariant to any tilt applied in the body frame, and its gradient with
/// respect to a body right-perturbation is exactly `yaw_axis_body(R)^T`.
///
/// Degenerate configuration: `R^T e_z = -e_z`, i.e. the device is rolled exactly
/// 180 degrees. There the yaw-free tilt is not unique and yaw is genuinely
/// ill-conditioned; [`rotation_aligning`] keeps it deterministic rather than
/// making it meaningful.
#[must_use]
pub fn yaw_of(attitude: &So3) -> Scalar {
    let tilt = rotation_aligning(&yaw_axis_body(attitude), &world_up());
    // M = R · T^-1 fixes e_z by construction, so it is a pure rotation about
    // world Z and its first column carries the yaw directly.
    let m = attitude.compose(&tilt.inverse()).matrix();
    m[(1, 0)].atan2(m[(0, 0)])
}

/// Gradient of [`yaw_of`] with respect to a body right-perturbation, as a
/// column (the measurement Jacobian is its transpose).
///
/// It is *not* [`yaw_axis_body`], and the difference is the subtle part of this
/// crate. Rotating about world `+Z` by `dψ` does change `yaw_of` by `dψ`, so the
/// gradient satisfies `g · v = 1`; but the two remaining directions are not
/// yaw-preserving once the device is tilted, because a product of two
/// horizontal-axis rotations is not itself a horizontal-axis rotation. That
/// holonomy is real geometry, not a modelling choice, and no definition of
/// "yaw" as a function of `R` escapes it.
///
/// Writing `R = Rz(ψ) · exp(η)` with `η` horizontal, the perturbations that
/// hold `ψ` fixed are `Jr(η) · (span e_x, e_y)`, so the gradient is normal to
/// that subspace: `g ∝ Jr(η)^-T e_z`, scaled to satisfy `g · v = 1`.
///
/// Using this rather than `v` keeps [`crate::OrientationFilter::correct_yaw`] a
/// consistent EKF update. It does not make the yaw correction spill into
/// roll/pitch in practice: the correction direction is `P g`, and with tilt
/// well determined and yaw barely determined, `P g` is almost exactly along the
/// yaw direction anyway. The covariance decides, which is the point of having
/// one.
#[must_use]
pub fn yaw_jacobian_body(attitude: &So3) -> Vec3 {
    let v = yaw_axis_body(attitude);
    let eta = rotation_aligning(&v, &world_up()).log();
    // Jr^-T = (Jr^T)^-1 = Jl^-1, since Jr(phi)^T = Jl(phi).
    let raw = So3::left_jacobian_inv(&eta) * world_up();
    let scale = raw.dot(&v);
    if scale.abs() < 1e-9 {
        // Only reachable at the upside-down singularity, where yaw is
        // ill-conditioned anyway; fall back to the direction that is at least
        // guaranteed to be a world-vertical rotation.
        return v;
    }
    raw / scale
}

/// Angle between the gravity directions of two attitudes: the roll/pitch error,
/// with yaw quotiented out.
///
/// This is the quantity spec.md §6 L1 calls *"roll/pitch error vs gravity"*.
/// Taking it as an angle between two vectors rather than as a difference of two
/// Euler pairs keeps it well defined at every attitude.
#[must_use]
pub fn tilt_between(a: &So3, b: &So3) -> Scalar {
    let (u, v) = (yaw_axis_body(a), yaw_axis_body(b));
    u.cross(&v).norm().atan2(u.dot(&v))
}

/// Wrap an angle into `(-pi, pi]`.
#[must_use]
pub fn wrap_angle(a: Scalar) -> Scalar {
    use std::f64::consts::{PI, TAU};
    let mut x = (a + PI).rem_euclid(TAU) - PI;
    if x <= -PI {
        x += TAU;
    }
    x
}

/// Normalise, or `None` if the vector is too short to have a direction.
fn unit_or_none(v: &Vec3) -> Option<Vec3> {
    let n = v.norm();
    if n.is_finite() && n > 1e-12 {
        Some(v / n)
    } else {
        None
    }
}

/// Some unit vector orthogonal to `a`. Crossing with the world axis least
/// aligned with `a` keeps the result well conditioned for every input.
fn any_orthogonal(a: &Vec3) -> Vec3 {
    let axis = if a.x.abs() <= a.y.abs() && a.x.abs() <= a.z.abs() {
        Vec3::x()
    } else if a.y.abs() <= a.z.abs() {
        Vec3::y()
    } else {
        Vec3::z()
    };
    a.cross(&axis).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::{FRAC_PI_2, PI};
    use wslam_core::math::Mat3;
    use wslam_core::DeterministicRng;

    fn rz(a: Scalar) -> So3 {
        So3::exp(&(Vec3::z() * a))
    }
    fn rx(a: Scalar) -> So3 {
        So3::exp(&(Vec3::x() * a))
    }

    #[test]
    fn rotation_aligning_takes_from_onto_to() {
        let mut rng = DeterministicRng::new("align", 1);
        for _ in 0..200 {
            let a = Vec3::new(rng.normal(), rng.normal(), rng.normal());
            let b = Vec3::new(rng.normal(), rng.normal(), rng.normal());
            if a.norm() < 1e-6 || b.norm() < 1e-6 {
                continue;
            }
            let r = rotation_aligning(&a, &b);
            assert_relative_eq!(r.act(&a.normalize()), b.normalize(), epsilon = 1e-12);
        }
    }

    #[test]
    fn rotation_aligning_handles_the_antiparallel_case() {
        // The degenerate branch: a phone rolled exactly 180 degrees. Identity
        // would be a 180-degree error, which is the bug this branch exists for.
        for from in [Vec3::z(), Vec3::x(), Vec3::new(0.3, -0.4, 0.5).normalize()] {
            let r = rotation_aligning(&from, &(-from));
            assert_relative_eq!(r.act(&from), -from, epsilon = 1e-12);
            assert_relative_eq!(r.angle(), PI, epsilon = 1e-9);
        }
    }

    #[test]
    fn rotation_aligning_is_identity_for_parallel_and_degenerate_input() {
        assert_relative_eq!(
            rotation_aligning(&Vec3::z(), &(Vec3::z() * 3.0)).matrix(),
            Mat3::identity(),
            epsilon = 1e-15
        );
        assert_relative_eq!(
            rotation_aligning(&Vec3::zeros(), &Vec3::z()).matrix(),
            Mat3::identity(),
            epsilon = 1e-15
        );
    }

    #[test]
    fn rotation_aligning_result_is_yaw_free() {
        // The minimal rotation onto world up has a horizontal axis, so it
        // contributes no yaw. Initialisation depends on this.
        let mut rng = DeterministicRng::new("align-yaw", 2);
        for _ in 0..100 {
            let up_body =
                Vec3::new(rng.normal(), rng.normal(), rng.normal().abs() + 0.1).normalize();
            let r = rotation_aligning(&up_body, &world_up());
            assert_relative_eq!(yaw_of(&r), 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn yaw_of_recovers_the_yaw_it_was_built_from() {
        // Round-trip: yaw_of(Rz(psi) * tilt) == psi for any yaw-free tilt.
        let mut rng = DeterministicRng::new("yaw-roundtrip", 3);
        for _ in 0..500 {
            let psi = rng.uniform_range(-PI + 1e-6, PI);
            let horizontal = Vec3::new(rng.normal(), rng.normal(), 0.0).normalize()
                * rng.uniform_range(0.0, 2.0);
            let tilt = So3::exp(&horizontal);
            assert_relative_eq!(yaw_of(&rz(psi).compose(&tilt)), psi, epsilon = 1e-9);
        }
    }

    #[test]
    fn yaw_of_matches_the_euler_form_when_level() {
        for psi in [-2.0, -0.3, 0.0, 0.7, 3.0] {
            assert_relative_eq!(yaw_of(&rz(psi)), psi, epsilon = 1e-12);
        }
    }

    #[test]
    fn yaw_of_is_invariant_to_body_tilt() {
        // A tilt applied in the body frame is by construction yaw-free, which is
        // the property that makes the gravity update leave yaw alone.
        let base = rz(0.9);
        for angle in [0.1, 0.5, 1.2, FRAC_PI_2] {
            assert_relative_eq!(yaw_of(&base.compose(&rx(angle))), 0.9, epsilon = 1e-9);
        }
    }

    #[test]
    fn yaw_jacobian_matches_central_differences() {
        let mut rng = DeterministicRng::new("yaw-jacobian", 5);
        for _ in 0..50 {
            let r = So3::exp(&Vec3::new(rng.normal(), rng.normal(), rng.normal()));
            // Stay clear of the upside-down singularity, where yaw is not a
            // well-conditioned function of R and no Jacobian is meaningful.
            if yaw_axis_body(&r).z < -0.9 {
                continue;
            }
            let g = yaw_jacobian_body(&r);
            let h = 1e-6;
            for i in 0..3 {
                let mut d = Vec3::zeros();
                d[i] = h;
                let num = (wrap_angle(yaw_of(&r.plus(&d)) - yaw_of(&r.plus(&(-d))))) / (2.0 * h);
                assert_relative_eq!(num, g[i], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn yaw_jacobian_reads_exactly_one_along_the_world_vertical() {
        // g . v == 1: a world yaw of dpsi changes yaw_of by dpsi, exactly. This
        // is the property that makes correct_yaw a heading correction.
        let mut rng = DeterministicRng::new("yaw-jacobian-norm", 6);
        for _ in 0..200 {
            let r = So3::exp(&Vec3::new(rng.normal(), rng.normal(), rng.normal()));
            if yaw_axis_body(&r).z < -0.9 {
                continue;
            }
            assert_relative_eq!(
                yaw_jacobian_body(&r).dot(&yaw_axis_body(&r)),
                1.0,
                epsilon = 1e-9
            );
        }
        // Level: the two coincide.
        assert_relative_eq!(yaw_jacobian_body(&rz(0.4)), Vec3::z(), epsilon = 1e-12);
    }

    #[test]
    fn yaw_axis_is_the_world_z_perturbation_direction() {
        // Rz(dpsi) * R == R * exp(dpsi * yaw_axis_body(R)) — the identity the
        // whole yaw treatment rests on.
        let r = So3::exp(&Vec3::new(0.3, 0.8, -0.2));
        let dpsi = 0.017;
        let lhs = rz(dpsi).compose(&r);
        let rhs = r.plus(&(yaw_axis_body(&r) * dpsi));
        assert_relative_eq!(lhs.matrix(), rhs.matrix(), epsilon = 1e-12);
    }

    #[test]
    fn gravity_direction_points_down_when_level() {
        assert_relative_eq!(
            gravity_direction_body(&So3::identity()),
            -Vec3::z(),
            epsilon = 1e-15
        );
        // Rolled 90 degrees about body x: world down lands on body -y.
        assert_relative_eq!(
            gravity_direction_body(&rx(FRAC_PI_2)),
            Vec3::new(0.0, -1.0, 0.0),
            epsilon = 1e-12
        );
        // And the accelerometer at rest reads its negation, +g along body +y.
        assert_relative_eq!(
            rx(FRAC_PI_2).inverse().act(&world_specific_force()),
            Vec3::new(0.0, wslam_core::imu::GRAVITY, 0.0),
            epsilon = 1e-12
        );
    }

    #[test]
    fn tilt_between_ignores_yaw_and_measures_tilt() {
        let a = rz(1.1);
        assert_relative_eq!(tilt_between(&a, &rz(-2.0)), 0.0, epsilon = 1e-12);
        assert_relative_eq!(
            tilt_between(&a, &a.compose(&rx(0.25))),
            0.25,
            epsilon = 1e-12
        );
    }

    #[test]
    fn wrap_angle_lands_in_the_half_open_interval() {
        use std::f64::consts::TAU;
        for k in -3..=3 {
            let base = 0.4 + k as f64 * TAU;
            assert_relative_eq!(wrap_angle(base), 0.4, epsilon = 1e-12);
        }
        assert_relative_eq!(wrap_angle(PI), PI, epsilon = 1e-12);
        assert_relative_eq!(wrap_angle(-PI), PI, epsilon = 1e-12);
        assert!(wrap_angle(3.5) < 0.0, "3.5 rad wraps to negative");
    }
}
