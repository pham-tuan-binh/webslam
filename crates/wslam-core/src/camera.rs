//! Camera intrinsics and lens distortion.
//!
//! The distortion model is not optional detail. spec.md §5 records that
//! Hayman & Murray (CVIU 2004) found *"barrel distortion produces a sharply
//! increasing overestimate of focal length, then outright failure"* for
//! rotation-based self-calibration, and §9 lists it as a risk with the
//! mitigation "distortion in model; ablation is a gate". So the model carries
//! distortion from the start, and L2 can switch it off only to run that
//! ablation deliberately.

use crate::math::{Mat3, Scalar, Vec2, Vec3};

/// Brown-Conrady radial-tangential distortion.
///
/// `x_d = x (1 + k1 r^2 + k2 r^4 + k3 r^6) + [2 p1 x y + p2 (r^2 + 2x^2),
///                                            p1 (r^2 + 2y^2) + 2 p2 x y]`
///
/// evaluated in normalised image coordinates. Phone wide cameras are strongly
/// barrel-distorted, so `k1` is typically negative and material.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RadialTangential {
    /// Second-order radial coefficient.
    pub k1: Scalar,
    /// Fourth-order radial coefficient.
    pub k2: Scalar,
    /// First tangential coefficient.
    pub p1: Scalar,
    /// Second tangential coefficient.
    pub p2: Scalar,
    /// Sixth-order radial coefficient.
    pub k3: Scalar,
}

impl RadialTangential {
    /// No distortion — the pinhole ablation arm.
    pub const NONE: Self = RadialTangential {
        k1: 0.0,
        k2: 0.0,
        p1: 0.0,
        p2: 0.0,
        k3: 0.0,
    };

    /// Radial-only model, the usual starting point for a phone camera.
    #[must_use]
    pub fn radial(k1: Scalar, k2: Scalar) -> Self {
        RadialTangential {
            k1,
            k2,
            ..Self::NONE
        }
    }

    /// Whether every coefficient is zero.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.k1 == 0.0 && self.k2 == 0.0 && self.p1 == 0.0 && self.p2 == 0.0 && self.k3 == 0.0
    }

    /// Apply distortion to a normalised image point.
    #[must_use]
    pub fn distort(&self, p: Vec2) -> Vec2 {
        if self.is_identity() {
            return p;
        }
        let (x, y) = (p.x, p.y);
        let r2 = x * x + y * y;
        let radial = 1.0 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3));
        let dx = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
        let dy = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;
        Vec2::new(x * radial + dx, y * radial + dy)
    }

    /// Jacobian of [`RadialTangential::distort`] with respect to the
    /// normalised input point.
    #[must_use]
    pub fn distort_jacobian(&self, p: Vec2) -> nalgebra::Matrix2<Scalar> {
        let (x, y) = (p.x, p.y);
        let r2 = x * x + y * y;
        let radial = 1.0 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3));
        // d(radial)/d(r2), then chain through d(r2)/dx = 2x.
        let d_radial_d_r2 = self.k1 + r2 * (2.0 * self.k2 + 3.0 * self.k3 * r2);
        let (dr_dx, dr_dy) = (2.0 * x * d_radial_d_r2, 2.0 * y * d_radial_d_r2);
        nalgebra::Matrix2::new(
            radial + x * dr_dx + 2.0 * self.p1 * y + 6.0 * self.p2 * x,
            x * dr_dy + 2.0 * self.p1 * x + 2.0 * self.p2 * y,
            y * dr_dx + 2.0 * self.p1 * x + 2.0 * self.p2 * y,
            radial + y * dr_dy + 6.0 * self.p1 * y + 2.0 * self.p2 * x,
        )
    }

    /// Invert distortion by Newton iteration on `distort(q) - p = 0`.
    ///
    /// Newton rather than the usual fixed point `q -= distort(q) - p`: at the
    /// image corner of a phone wide lens (`k1 ~ -0.3`, `r ~ 0.9`) the fixed
    /// point converges linearly and still carries ~1e-8 of residual after a
    /// dozen iterations, which is a visible fraction of a pixel once multiplied
    /// by the focal length. Newton clears it in four, and this runs once per
    /// feature per frame.
    ///
    /// Falls back to a fixed-point step if the Jacobian is singular, and
    /// returns the best iterate rather than failing — the caller cannot do
    /// anything useful with a failure.
    #[must_use]
    pub fn undistort(&self, p: Vec2) -> Vec2 {
        if self.is_identity() {
            return p;
        }
        let mut q = p;
        for _ in 0..10 {
            let err = self.distort(q) - p;
            if err.norm_squared() < 1e-28 {
                break;
            }
            match self.distort_jacobian(q).try_inverse() {
                Some(inv) => q -= inv * err,
                None => q -= err,
            }
        }
        q
    }
}

/// Pinhole intrinsics plus distortion.
///
/// `fx`/`fy` are separate even though phone sensors are square-pixel, because
/// the L2 linear self-calibration methods in spec.md §5 assume zero skew *or*
/// square pixels and we want to be able to test that assumption rather than
/// bake it in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraIntrinsics {
    /// Focal length in pixels, x.
    pub fx: Scalar,
    /// Focal length in pixels, y.
    pub fy: Scalar,
    /// Principal point x, in pixels.
    pub cx: Scalar,
    /// Principal point y, in pixels.
    pub cy: Scalar,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Lens distortion.
    pub distortion: RadialTangential,
}

impl CameraIntrinsics {
    /// Construct from a focal length and image size, assuming a centred
    /// principal point and square pixels. This is the prior L2 refines.
    #[must_use]
    pub fn from_focal(focal_px: Scalar, width: u32, height: u32) -> Self {
        CameraIntrinsics {
            fx: focal_px,
            fy: focal_px,
            cx: width as Scalar * 0.5,
            cy: height as Scalar * 0.5,
            width,
            height,
            distortion: RadialTangential::NONE,
        }
    }

    /// Construct from a horizontal field of view in degrees.
    ///
    /// The default prior when nothing else is known: phone rear cameras cluster
    /// near 65-70 degrees horizontal.
    #[must_use]
    pub fn from_hfov_degrees(hfov_deg: Scalar, width: u32, height: u32) -> Self {
        let half = (hfov_deg.to_radians() * 0.5).tan();
        Self::from_focal(width as Scalar * 0.5 / half, width, height)
    }

    /// The 3x3 calibration matrix `K`.
    #[must_use]
    pub fn matrix(&self) -> Mat3 {
        Mat3::new(self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0)
    }

    /// `K^-1`.
    #[must_use]
    pub fn inverse_matrix(&self) -> Mat3 {
        Mat3::new(
            1.0 / self.fx,
            0.0,
            -self.cx / self.fx,
            0.0,
            1.0 / self.fy,
            -self.cy / self.fy,
            0.0,
            0.0,
            1.0,
        )
    }

    /// Project a point in **camera** coordinates to pixels, applying distortion.
    ///
    /// Returns `None` for points at or behind the image plane, which callers
    /// must handle — a point behind the camera projects to a perfectly
    /// plausible pixel if you forget to check the sign of `z`.
    #[must_use]
    pub fn project(&self, p_cam: &Vec3) -> Option<Vec2> {
        if p_cam.z <= 1e-9 {
            return None;
        }
        let n = Vec2::new(p_cam.x / p_cam.z, p_cam.y / p_cam.z);
        let d = self.distortion.distort(n);
        Some(Vec2::new(self.fx * d.x + self.cx, self.fy * d.y + self.cy))
    }

    /// Project without the `z > 0` check, for Jacobian evaluation where the
    /// caller has already validated depth.
    #[must_use]
    pub fn project_unchecked(&self, p_cam: &Vec3) -> Vec2 {
        let inv_z = 1.0 / p_cam.z;
        let n = Vec2::new(p_cam.x * inv_z, p_cam.y * inv_z);
        let d = self.distortion.distort(n);
        Vec2::new(self.fx * d.x + self.cx, self.fy * d.y + self.cy)
    }

    /// Pixel -> normalised undistorted image coordinates.
    #[must_use]
    pub fn unproject_normalized(&self, px: Vec2) -> Vec2 {
        self.distortion.undistort(Vec2::new(
            (px.x - self.cx) / self.fx,
            (px.y - self.cy) / self.fy,
        ))
    }

    /// Pixel -> unit bearing vector in camera coordinates.
    #[must_use]
    pub fn unproject_bearing(&self, px: Vec2) -> Vec3 {
        let n = self.unproject_normalized(px);
        Vec3::new(n.x, n.y, 1.0).normalize()
    }

    /// Jacobian of the *pinhole* projection with respect to the camera-frame
    /// point, `d(pixel)/d(p_cam)`, ignoring distortion.
    ///
    /// Distortion is deliberately excluded: tracking undistorts features once on
    /// extraction and then works in the pinhole model, which keeps the PnP and
    /// BA Jacobians exact rather than approximate.
    #[must_use]
    pub fn projection_jacobian(&self, p_cam: &Vec3) -> nalgebra::Matrix2x3<Scalar> {
        let inv_z = 1.0 / p_cam.z;
        let inv_z2 = inv_z * inv_z;
        nalgebra::Matrix2x3::new(
            self.fx * inv_z,
            0.0,
            -self.fx * p_cam.x * inv_z2,
            0.0,
            self.fy * inv_z,
            -self.fy * p_cam.y * inv_z2,
        )
    }

    /// Horizontal field of view in degrees.
    #[must_use]
    pub fn hfov_degrees(&self) -> Scalar {
        2.0 * (self.width as Scalar * 0.5 / self.fx).atan().to_degrees()
    }

    /// Whether a pixel lies inside the image, with an optional margin.
    #[must_use]
    pub fn contains(&self, px: Vec2, margin: Scalar) -> bool {
        px.x >= margin
            && px.y >= margin
            && px.x < self.width as Scalar - margin
            && px.y < self.height as Scalar - margin
    }

    /// Intrinsics for an image downscaled by `factor` in each dimension.
    #[must_use]
    pub fn scaled(&self, factor: Scalar) -> Self {
        CameraIntrinsics {
            fx: self.fx * factor,
            fy: self.fy * factor,
            // Pixel centres, not corners: a 2x downscale maps pixel centre
            // (x + 0.5) to (x + 0.5) / 2, hence the half-pixel correction.
            cx: (self.cx + 0.5) * factor - 0.5,
            cy: (self.cy + 0.5) * factor - 0.5,
            width: ((self.width as Scalar * factor).round() as u32).max(1),
            height: ((self.height as Scalar * factor).round() as u32).max(1),
            distortion: self.distortion,
        }
    }
}

/// Which camera model a pose stream was produced under. Recorded in map
/// metadata so a map anchored with one lens is not silently reused with another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraModel {
    /// Pinhole, no distortion terms.
    Pinhole,
    /// Pinhole plus Brown-Conrady radial-tangential.
    RadialTangential,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn project_unproject_roundtrip_pinhole() {
        let k = CameraIntrinsics::from_focal(600.0, 640, 480);
        let p = Vec3::new(0.3, -0.2, 2.0);
        let px = k.project(&p).unwrap();
        let bearing = k.unproject_bearing(px);
        assert_relative_eq!(bearing, p.normalize(), epsilon = 1e-12);
    }

    #[test]
    fn project_rejects_points_behind_camera() {
        let k = CameraIntrinsics::from_focal(600.0, 640, 480);
        assert!(k.project(&Vec3::new(0.1, 0.1, -1.0)).is_none());
        assert!(k.project(&Vec3::new(0.1, 0.1, 0.0)).is_none());
    }

    #[test]
    fn distortion_roundtrip_at_phone_scale() {
        // k1 = -0.28 is a realistic phone wide-angle barrel coefficient.
        let d = RadialTangential {
            k1: -0.28,
            k2: 0.09,
            p1: 0.001,
            p2: -0.0008,
            k3: -0.01,
        };
        for &(x, y) in &[(0.0, 0.0), (0.2, -0.1), (-0.5, 0.4), (0.7, 0.7)] {
            let p = Vec2::new(x, y);
            assert_relative_eq!(d.undistort(d.distort(p)), p, epsilon = 1e-13);
        }
    }

    #[test]
    fn distort_jacobian_matches_numerical() {
        let d = RadialTangential {
            k1: -0.28,
            k2: 0.09,
            p1: 0.001,
            p2: -0.0008,
            k3: -0.01,
        };
        let p = Vec2::new(0.35, -0.5);
        let j = d.distort_jacobian(p);
        let eps = 1e-7;
        for i in 0..2 {
            let mut dp = Vec2::zeros();
            dp[i] = eps;
            let num = (d.distort(p + dp) - d.distort(p - dp)) / (2.0 * eps);
            assert_relative_eq!(j[(0, i)], num.x, epsilon = 1e-6);
            assert_relative_eq!(j[(1, i)], num.y, epsilon = 1e-6);
        }
    }

    #[test]
    fn barrel_distortion_pulls_points_inward() {
        let d = RadialTangential::radial(-0.3, 0.0);
        let p = Vec2::new(0.6, 0.0);
        assert!(d.distort(p).x < p.x, "negative k1 must compress radially");
    }

    #[test]
    fn projection_jacobian_matches_numerical() {
        let k = CameraIntrinsics::from_focal(517.3, 640, 480);
        let p = Vec3::new(0.4, -0.3, 3.1);
        let j = k.projection_jacobian(&p);
        let eps = 1e-7;
        for i in 0..3 {
            let mut dp = Vec3::zeros();
            dp[i] = eps;
            let num =
                (k.project_unchecked(&(p + dp)) - k.project_unchecked(&(p - dp))) / (2.0 * eps);
            assert_relative_eq!(j[(0, i)], num.x, epsilon = 1e-4);
            assert_relative_eq!(j[(1, i)], num.y, epsilon = 1e-4);
        }
    }

    #[test]
    fn hfov_roundtrip() {
        let k = CameraIntrinsics::from_hfov_degrees(66.0, 1280, 720);
        assert_relative_eq!(k.hfov_degrees(), 66.0, epsilon = 1e-9);
    }

    #[test]
    fn pyramid_scaling_preserves_principal_point_geometry() {
        let k = CameraIntrinsics::from_focal(600.0, 640, 480);
        let half = k.scaled(0.5);
        assert_eq!((half.width, half.height), (320, 240));
        assert_relative_eq!(half.fx, 300.0, epsilon = 1e-12);
        // Centre pixel of the full image maps to centre pixel of the half image.
        assert_relative_eq!(half.cx, (320.0 + 0.5) * 0.5 - 0.5, epsilon = 1e-12);
    }

    #[test]
    fn k_inverse_is_inverse() {
        let k = CameraIntrinsics {
            fx: 512.0,
            fy: 511.0,
            cx: 319.5,
            cy: 241.0,
            width: 640,
            height: 480,
            distortion: RadialTangential::NONE,
        };
        assert_relative_eq!(
            k.matrix() * k.inverse_matrix(),
            Mat3::identity(),
            epsilon = 1e-12
        );
    }
}
