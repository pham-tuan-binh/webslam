//! Covariance ellipsoids.
//!
//! spec.md §8 lists *"covariance ellipsoid on the current pose"* as a required
//! viewer element: *"it is our differentiator, we should be looking at it
//! daily."* Turning a 6x6 covariance into something drawable is this file.

use wslam_core::{Mat3, Mat6, Scalar, So3, Vec3};

/// An uncertainty ellipsoid: three semi-axis lengths and their orientation.
#[derive(Debug, Clone, Copy)]
pub struct Ellipsoid {
    /// Semi-axis lengths, in the units of the block that produced them —
    /// metres for the translation block, radians for the rotation block.
    pub semi_axes: Vec3,
    /// Rotation taking ellipsoid-local axes to the parent frame.
    pub orientation: So3,
    /// The sigma multiple these axes represent.
    pub sigma: Scalar,
}

impl Ellipsoid {
    /// Volume. A single scalar for "how uncertain is this", useful as a time
    /// series when the full ellipsoid is too much to look at.
    #[must_use]
    pub fn volume(&self) -> Scalar {
        (4.0 / 3.0) * std::f64::consts::PI * self.semi_axes.x * self.semi_axes.y * self.semi_axes.z
    }

    /// Largest semi-axis — the worst-case direction.
    #[must_use]
    pub fn max_extent(&self) -> Scalar {
        self.semi_axes.max()
    }
}

/// Extract a drawable ellipsoid from a 3x3 block of a pose covariance.
///
/// `block_offset` is 0 for the translation block and 3 for the rotation block,
/// matching the `[translation, rotation]` ordering used throughout.
///
/// Returns `None` when the block is not finite — which is the honest outcome
/// for a pose that has not converged, and better than drawing an ellipsoid the
/// size of the scene.
#[must_use]
pub fn uncertainty_ellipsoid(
    covariance: &Mat6,
    block_offset: usize,
    sigma: Scalar,
) -> Option<Ellipsoid> {
    debug_assert!(block_offset == 0 || block_offset == 3);
    let mut block = Mat3::zeros();
    for r in 0..3 {
        for c in 0..3 {
            let v = covariance[(block_offset + r, block_offset + c)];
            if !v.is_finite() {
                return None;
            }
            block[(r, c)] = v;
        }
    }
    // Symmetrise before decomposing: filter updates drift out of symmetry by a
    // few ulps, and `symmetric_eigen` on an asymmetric input silently uses only
    // the lower triangle, which would quietly draw the wrong shape.
    let symmetric = (block + block.transpose()) * 0.5;
    let eigen = symmetric.symmetric_eigen();

    let mut axes = Vec3::zeros();
    for i in 0..3 {
        // Clamp at zero rather than returning None: a marginally negative
        // eigenvalue is numerical noise around a genuinely tiny variance, and
        // refusing to draw would hide a perfectly good pose.
        axes[i] = sigma * eigen.eigenvalues[i].max(0.0).sqrt();
    }

    // `symmetric_eigen` does not guarantee a right-handed basis, and a
    // left-handed one renders as a mirrored ellipsoid — which, being an
    // ellipsoid, looks completely fine and is wrong.
    let mut vectors = eigen.eigenvectors;
    if vectors.determinant() < 0.0 {
        vectors.set_column(2, &(-vectors.column(2)));
    }

    Some(Ellipsoid {
        semi_axes: axes,
        orientation: So3::from_matrix(&vectors),
        sigma,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn diagonal(values: [Scalar; 6]) -> Mat6 {
        Mat6::from_diagonal(&wslam_core::Vec6::from_row_slice(&values))
    }

    #[test]
    fn diagonal_covariance_gives_axis_aligned_sigmas() {
        let cov = diagonal([0.04, 0.01, 0.09, 1.0, 1.0, 1.0]);
        let e = uncertainty_ellipsoid(&cov, 0, 1.0).unwrap();
        let mut axes: Vec<Scalar> = e.semi_axes.iter().copied().collect();
        axes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_relative_eq!(axes[0], 0.1, epsilon = 1e-12);
        assert_relative_eq!(axes[1], 0.2, epsilon = 1e-12);
        assert_relative_eq!(axes[2], 0.3, epsilon = 1e-12);
    }

    #[test]
    fn sigma_scales_the_axes_linearly() {
        let cov = diagonal([0.25, 0.25, 0.25, 1.0, 1.0, 1.0]);
        let one = uncertainty_ellipsoid(&cov, 0, 1.0).unwrap();
        let three = uncertainty_ellipsoid(&cov, 0, 3.0).unwrap();
        assert_relative_eq!(three.max_extent(), 3.0 * one.max_extent(), epsilon = 1e-12);
    }

    #[test]
    fn rotation_block_is_selectable() {
        let cov = diagonal([1.0, 1.0, 1.0, 0.0004, 0.0004, 0.0004]);
        let e = uncertainty_ellipsoid(&cov, 3, 1.0).unwrap();
        assert_relative_eq!(e.max_extent(), 0.02, epsilon = 1e-12);
    }

    #[test]
    fn correlated_covariance_rotates_the_ellipsoid() {
        // A 45-degree correlated 2D block: the principal axes must come out
        // along the diagonals, not along x and y.
        let mut cov = Mat6::identity();
        cov[(0, 0)] = 0.05;
        cov[(1, 1)] = 0.05;
        cov[(0, 1)] = 0.04;
        cov[(1, 0)] = 0.04;
        cov[(2, 2)] = 0.01;
        let e = uncertainty_ellipsoid(&cov, 0, 1.0).unwrap();
        // Eigenvalues 0.09 and 0.01 -> semi-axes 0.3 and 0.1.
        let mut axes: Vec<Scalar> = e.semi_axes.iter().copied().collect();
        axes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_relative_eq!(axes[2], 0.3, epsilon = 1e-9);
        // And the basis must be a genuine rotation, not a reflection.
        assert_relative_eq!(e.orientation.matrix().determinant(), 1.0, epsilon = 1e-9);
    }

    #[test]
    fn basis_is_always_right_handed() {
        // A left-handed basis renders as a mirrored ellipsoid, which looks
        // entirely plausible and is wrong.
        for scale in [1.0, 3.0, 0.001] {
            let cov = diagonal([0.04 * scale, 0.09 * scale, 0.01 * scale, 1.0, 1.0, 1.0]);
            let e = uncertainty_ellipsoid(&cov, 0, 1.0).unwrap();
            assert!(e.orientation.matrix().determinant() > 0.0);
        }
    }

    #[test]
    fn infinite_covariance_declines_rather_than_drawing_the_universe() {
        let cov = Mat6::identity() * Scalar::INFINITY;
        assert!(uncertainty_ellipsoid(&cov, 0, 1.0).is_none());
        let mut nan = Mat6::identity();
        nan[(1, 1)] = Scalar::NAN;
        assert!(uncertainty_ellipsoid(&nan, 0, 1.0).is_none());
    }

    #[test]
    fn marginally_negative_eigenvalues_are_clamped_not_rejected() {
        // Numerical noise around a genuinely tiny variance. Refusing to draw
        // here would hide a perfectly good pose.
        let cov = diagonal([-1e-18, 0.01, 0.01, 1.0, 1.0, 1.0]);
        let e = uncertainty_ellipsoid(&cov, 0, 1.0).unwrap();
        assert!(e.semi_axes.iter().all(|v| v.is_finite() && *v >= 0.0));
    }

    #[test]
    fn volume_matches_the_closed_form() {
        let cov = diagonal([0.04, 0.09, 0.16, 1.0, 1.0, 1.0]);
        let e = uncertainty_ellipsoid(&cov, 0, 1.0).unwrap();
        let expected = (4.0 / 3.0) * std::f64::consts::PI * 0.2 * 0.3 * 0.4;
        assert_relative_eq!(e.volume(), expected, epsilon = 1e-12);
    }
}
