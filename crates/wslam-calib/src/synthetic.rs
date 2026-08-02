//! Synthetic rotating rig used by the Tier-1 tests (spec.md §6: *"Everything
//! with a closed-form answer, on synthetic input"*).
//!
//! Test-only. It exists so every test in this crate knows the true focal
//! length, the true distortion and the true lever arm, and can therefore assert
//! recovery of a known answer rather than self-consistency of the estimator.
//!
//! The generative path deliberately runs through `wslam_core::CameraIntrinsics`
//! rather than reimplementing projection: if the two disagreed, the tests would
//! be validating the wrong camera model.

use wslam_core::{
    CameraIntrinsics, DeterministicRng, Mat3, RadialTangential, Scalar, So3, Vec2, Vec3,
};

/// A camera on a rotating rig, optionally with barrel distortion and a wrist
/// lever arm.
#[derive(Debug, Clone)]
pub struct SyntheticRig {
    /// True focal length in pixels, square pixels, centred principal point.
    pub focal: Scalar,
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// True lens distortion.
    pub distortion: RadialTangential,
    /// Pivot in camera coordinates, or `None` for rotation about the optical
    /// centre. `Some` injects `t = (I - R) l` exactly as Ji et al. describe.
    pub lever_arm: Option<Vec3>,
    /// Mean scene depth in metres.
    pub depth: Scalar,
    /// Fractional half-width of the uniform depth spread about `depth`.
    pub depth_spread: Scalar,
    /// Gaussian pixel noise standard deviation added to both views.
    pub noise_px: Scalar,
    /// Border kept clear when sampling, in pixels.
    pub margin_px: Scalar,
}

impl SyntheticRig {
    /// A distortion-free rig rotating exactly about its optical centre.
    pub fn pinhole(focal: Scalar, width: u32, height: u32) -> Self {
        SyntheticRig {
            focal,
            width,
            height,
            distortion: RadialTangential::NONE,
            lever_arm: None,
            depth: 3.0,
            depth_spread: 0.1,
            noise_px: 0.0,
            margin_px: 12.0,
        }
    }

    /// The ground-truth intrinsics.
    pub fn intrinsics(&self) -> CameraIntrinsics {
        let mut k = CameraIntrinsics::from_focal(self.focal, self.width, self.height);
        k.distortion = self.distortion;
        k
    }

    /// The principal point the estimator will assume.
    pub fn principal(&self) -> Vec2 {
        Vec2::new(self.width as Scalar * 0.5, self.height as Scalar * 0.5)
    }

    /// Generate `n` correspondences in raw pixel coordinates for the relative
    /// rotation `rotation` (`R_cam2_cam1`).
    ///
    /// Points are sampled by picking a pixel in frame 1, back-projecting it to
    /// a ray with the *true* intrinsics, assigning it a depth, transforming and
    /// reprojecting. Sampling in the image rather than in the world guarantees
    /// coverage out to the corners, which is exactly where the distortion
    /// signal lives.
    pub fn raw_pair(
        &self,
        rotation: &So3,
        n: usize,
        rng: &mut DeterministicRng,
    ) -> Vec<(Vec2, Vec2)> {
        let k = self.intrinsics();
        let r = rotation.matrix();
        let t = match self.lever_arm {
            Some(l) => (Mat3::identity() - r) * l,
            None => Vec3::zeros(),
        };
        let mut out = Vec::with_capacity(n);
        let mut attempts = 0usize;
        while out.len() < n && attempts < 80 * n {
            attempts += 1;
            let p1 = Vec2::new(
                rng.uniform_range(self.margin_px, self.width as Scalar - self.margin_px),
                rng.uniform_range(self.margin_px, self.height as Scalar - self.margin_px),
            );
            let nrm = k.unproject_normalized(p1);
            let z = self.depth * (1.0 + self.depth_spread * rng.uniform_range(-1.0, 1.0));
            let x1 = Vec3::new(nrm.x * z, nrm.y * z, z);
            let x2 = r * x1 + t;
            let Some(p2) = k.project(&x2) else { continue };
            if !k.contains(p2, self.margin_px) {
                continue;
            }
            let noisy = |p: Vec2, rng: &mut DeterministicRng| {
                if self.noise_px > 0.0 {
                    Vec2::new(
                        p.x + rng.normal_with(0.0, self.noise_px),
                        p.y + rng.normal_with(0.0, self.noise_px),
                    )
                } else {
                    p
                }
            };
            let a = noisy(p1, rng);
            let b = noisy(p2, rng);
            out.push((a, b));
        }
        out
    }

    /// [`SyntheticRig::raw_pair`], shifted to principal-point-centred pixels —
    /// the convention [`crate::focal_from_rotation_homography`] requires.
    pub fn centred_pair(
        &self,
        rotation: &So3,
        n: usize,
        rng: &mut DeterministicRng,
    ) -> Vec<(Vec2, Vec2)> {
        let c = self.principal();
        self.raw_pair(rotation, n, rng)
            .into_iter()
            .map(|(a, b)| (a - c, b - c))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn pure_rotation_pair_is_an_exact_infinite_homography() {
        // The generator must satisfy x2 ~ K R K^-1 x1 to machine precision,
        // otherwise every recovery test downstream is measuring generator bugs.
        let rig = SyntheticRig::pinhole(985.0, 1280, 720);
        let rotation = So3::exp(&Vec3::new(0.01, 0.05, -0.02));
        let mut rng = DeterministicRng::new("t", 1);
        let matches = rig.centred_pair(&rotation, 60, &mut rng);
        assert!(matches.len() > 40);

        let f = rig.focal;
        let k = Mat3::new(f, 0.0, 0.0, 0.0, f, 0.0, 0.0, 0.0, 1.0);
        let k_inv = Mat3::new(1.0 / f, 0.0, 0.0, 0.0, 1.0 / f, 0.0, 0.0, 0.0, 1.0);
        let h = k * rotation.matrix() * k_inv;
        for (a, b) in matches {
            let mapped = crate::homography::apply_homography(&h, a).unwrap();
            assert_relative_eq!(mapped, b, epsilon = 1e-9);
        }
    }

    #[test]
    fn depth_only_matters_when_there_is_a_lever_arm() {
        let rotation = So3::exp(&Vec3::new(0.0, 0.05, 0.0));
        let mut near = SyntheticRig::pinhole(985.0, 1280, 720);
        near.depth_spread = 0.0;
        near.depth = 0.5;
        let mut far = near.clone();
        far.depth = 50.0;
        let a = near.raw_pair(&rotation, 30, &mut DeterministicRng::new("t", 2));
        let b = far.raw_pair(&rotation, 30, &mut DeterministicRng::new("t", 2));
        for (x, y) in a.iter().zip(b.iter()) {
            assert_relative_eq!(x.1, y.1, epsilon = 1e-9);
        }

        // With a lever arm the same rotation produces depth-dependent parallax.
        let mut lever_near = near.clone();
        lever_near.lever_arm = Some(Vec3::new(0.0, 0.0, -0.20));
        let mut lever_far = lever_near.clone();
        lever_far.depth = 50.0;
        let c = lever_near.raw_pair(&rotation, 30, &mut DeterministicRng::new("t", 2));
        let d = lever_far.raw_pair(&rotation, 30, &mut DeterministicRng::new("t", 2));
        let moved = c
            .iter()
            .zip(d.iter())
            .map(|(x, y)| (x.1 - y.1).norm())
            .fold(0.0, Scalar::max);
        assert!(
            moved > 20.0,
            "lever-arm parallax at 0.5 m was only {moved} px"
        );
    }

    #[test]
    fn barrel_distortion_pulls_the_corners_in() {
        let mut rig = SyntheticRig::pinhole(985.0, 1280, 720);
        rig.distortion = RadialTangential::radial(-0.28, 0.0);
        let k = rig.intrinsics();
        let corner_ray = Vec3::new(0.6, 0.35, 1.0);
        let distorted = k.project(&corner_ray).unwrap();
        let pinhole = CameraIntrinsics::from_focal(985.0, 1280, 720)
            .project(&corner_ray)
            .unwrap();
        let c = rig.principal();
        assert!((distorted - c).norm() < (pinhole - c).norm());
    }
}
