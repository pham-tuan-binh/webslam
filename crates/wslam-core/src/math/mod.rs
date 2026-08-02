//! Lie group algebra for SO(3), SE(3) and Sim(3).
//!
//! Hand-rolled on `nalgebra` rather than pulled from `sophus-rs` (spec.md §7
//! names it as a candidate). Rationale in `docs/DECISIONS.md`: these operations
//! are the subject of the Tier-1 test suite, they are ~400 lines, and owning
//! them removes a dependency whose API churn would land squarely on the code we
//! most need to keep stable.
//!
//! ## Conventions, fixed once
//!
//! - Poses are **`T_world_camera`**: they map a point in camera coordinates to
//!   world coordinates. `pose.translation()` is therefore the camera centre in
//!   world coordinates.
//! - A twist is `[rho; phi]` — **translation first, rotation second** — matching
//!   the 6x6 covariance block order promised by the public `Pose` type
//!   (spec.md §3: `covariance: Float64Array; // 6x6, [translation, rotation]`).
//! - Perturbations are **right-multiplied**: `T ⊞ δ = T · exp(δ)`. Jacobians
//!   returned by this module are right Jacobians unless the name says otherwise.
//! - Matrices are column-major on export (`to_matrix_f32`), because that is what
//!   WebGL/WebGPU and three.js consume directly.

mod se3;
mod sim3;
mod so3;

pub use se3::Se3;
pub use sim3::{umeyama, Alignment, Sim3};
pub use so3::So3;

/// Scalar type used for all geometry. Image data is `u8`/`f32`; geometry is f64.
pub type Scalar = f64;

/// 2-vector (pixels, normalised image coordinates).
pub type Vec2 = nalgebra::Vector2<Scalar>;
/// 3-vector (metres, or up-to-scale units when `ScaleKind::None`).
pub type Vec3 = nalgebra::Vector3<Scalar>;
/// 6-vector twist, ordered `[rho; phi]` = `[translation; rotation]`.
pub type Vec6 = nalgebra::Vector6<Scalar>;
/// 3x3 matrix.
pub type Mat3 = nalgebra::Matrix3<Scalar>;
/// 4x4 homogeneous matrix.
pub type Mat4 = nalgebra::Matrix4<Scalar>;
/// 6x6 matrix, used for pose covariance in `[translation, rotation]` order.
pub type Mat6 = nalgebra::Matrix6<Scalar>;
/// Unit quaternion.
pub type Quat = nalgebra::UnitQuaternion<Scalar>;

/// Numerical threshold below which small-angle series expansions are used
/// instead of the closed forms, which lose precision as `theta -> 0`.
pub const SMALL_ANGLE: Scalar = 1e-8;

/// Skew-symmetric ("hat") operator: `hat(v) * w == v.cross(&w)`.
#[inline]
#[must_use]
pub fn hat(v: &Vec3) -> Mat3 {
    Mat3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// Inverse of [`hat`]. Takes the skew part; does not verify skew-symmetry.
#[inline]
#[must_use]
pub fn vee(m: &Mat3) -> Vec3 {
    Vec3::new(
        0.5 * (m[(2, 1)] - m[(1, 2)]),
        0.5 * (m[(0, 2)] - m[(2, 0)]),
        0.5 * (m[(1, 0)] - m[(0, 1)]),
    )
}

/// Split a `[rho; phi]` twist into its translation and rotation halves.
#[inline]
#[must_use]
pub fn split_twist(xi: &Vec6) -> (Vec3, Vec3) {
    (
        Vec3::new(xi[0], xi[1], xi[2]),
        Vec3::new(xi[3], xi[4], xi[5]),
    )
}

/// Join translation and rotation halves into a `[rho; phi]` twist.
#[inline]
#[must_use]
pub fn join_twist(rho: &Vec3, phi: &Vec3) -> Vec6 {
    Vec6::new(rho.x, rho.y, rho.z, phi.x, phi.y, phi.z)
}
