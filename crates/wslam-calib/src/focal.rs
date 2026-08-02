//! The infinite-homography focal solve, and robust aggregation across pairs.
//!
//! For a camera rotating about its optical centre, `x2 ~ K R K^-1 x1`
//! (Hartley, ECCV 1994; de Agapito, Hayman & Reid, IJCV 45(2), 2001). L1 hands
//! us `R` from the gyro, so the only unknown left in the infinite homography is
//! `K`, and under the zero-skew / square-pixel assumption spec.md §5 names, `K`
//! is one number.

use wslam_core::stats::{mad_sigma, median};
use wslam_core::{Mat3, Scalar, So3};

use crate::homography::normalize_homography;

/// Minimum `sin` of the angle between the two optical axes for the focal length
/// to be observable.
///
/// The focal length enters the infinite homography only through the third row
/// and third column, whose magnitudes are both `sqrt(1 - r22^2)` — the sine of
/// the angle the optical axis swept. Two degenerate motions collapse it to
/// zero: no rotation at all, and rotation *about* the optical axis, where the
/// image simply spins and carries no information about focal length whatever.
/// `0.02` is `1.15` degrees; below it the noise gain on `f` exceeds a factor of
/// fifty and the answer is decoration.
pub const MIN_AXIS_TILT: Scalar = 0.02;

/// Sine of the angle between the two optical axes — the quantity that has to be
/// non-zero for [`focal_from_rotation_homography`] to have an answer.
#[must_use]
pub fn axis_tilt(r: &So3) -> Scalar {
    let r22 = r.matrix()[(2, 2)];
    (1.0 - r22 * r22).max(0.0).sqrt()
}

/// Whether a relative rotation carries enough optical-axis motion to identify
/// the focal length. See [`MIN_AXIS_TILT`].
#[must_use]
pub fn rotation_is_observable(r: &So3) -> bool {
    axis_tilt(r) >= MIN_AXIS_TILT
}

/// Focal length in pixels from one infinite homography and its known rotation.
///
/// `h` maps **principal-point-centred** pixels of frame 1 onto frame 2, and `r`
/// is `R_cam2_cam1` — the rotation taking a direction expressed in frame 1 into
/// frame 2. Under `K = diag(f, f, 1)` the constraint `H = lambda K R K^-1`
/// expands to
///
/// ```text
/// H = lambda * [ r00      r01      f*r02 ]
///              [ r10      r11      f*r12 ]
///              [ r20/f    r21/f    r22   ]
/// ```
///
/// Least squares on the third column gives `lambda*f`, on the third row gives
/// `lambda/f`, and the two share the identical denominator `1 - r22^2`, so the
/// ratio is
///
/// ```text
/// f^2 = (h02*r02 + h12*r12) / (h20*r20 + h21*r21)
/// ```
///
/// The unknown projective scale `lambda` divides out, which is why this form is
/// immune to the arbitrary scale *and sign* the DLT returns.
///
/// Returns `None` for the degenerate rotations described at [`MIN_AXIS_TILT`],
/// and when the ratio is non-positive — a negative `f^2` means the data cannot
/// be explained by a rotating pinhole at any focal length, and inventing one
/// would be worse than admitting it.
#[must_use]
pub fn focal_from_rotation_homography(h: &Mat3, r: &So3) -> Option<Scalar> {
    let rm = r.matrix();
    if !rotation_is_observable(r) {
        return None;
    }
    // Scale-normalise so the two accumulations are comparable in magnitude;
    // the ratio is scale-free but the finite-precision sum is not.
    let h = normalize_homography(h);

    let column = h[(0, 2)] * rm[(0, 2)] + h[(1, 2)] * rm[(1, 2)];
    let row = h[(2, 0)] * rm[(2, 0)] + h[(2, 1)] * rm[(2, 1)];
    if !row.is_finite() || !column.is_finite() || row.abs() < 1e-18 {
        return None;
    }
    let f_squared = column / row;
    if !(f_squared > 0.0) || !f_squared.is_finite() {
        return None;
    }
    let f = f_squared.sqrt();
    if !f.is_finite() || f <= 0.0 {
        return None;
    }
    Some(f)
}

/// Robust summary of the per-pair focal estimates.
#[derive(Debug, Clone)]
pub struct FocalAggregate {
    /// Robust centre, in pixels.
    pub focal_px: Scalar,
    /// Variance of [`FocalAggregate::focal_px`], in px^2.
    pub variance: Scalar,
    /// Robust spread of the individual samples, in pixels.
    pub sigma: Scalar,
    /// Indices of the samples that survived outlier rejection.
    pub kept: Vec<usize>,
}

/// Relative sigma assumed when a single pair leaves no spread to measure.
///
/// One homography gives no handle on its own dispersion. Reporting zero
/// variance there would be the exact overconfidence spec.md §6 L6 calls "worse
/// than no covariance at all", so we fall back to the order the rotating-camera
/// literature reports for a single pair.
const SINGLE_SAMPLE_RELATIVE_SIGMA: Scalar = 0.05;

/// Multiple of the robust sigma beyond which a per-pair focal is discarded.
const OUTLIER_SIGMAS: Scalar = 3.0;

/// Efficiency factor of the median relative to the mean for Gaussian data:
/// `var(median) = (pi/2) * sigma^2 / n`.
const MEDIAN_VARIANCE_FACTOR: Scalar = std::f64::consts::FRAC_PI_2;

/// Aggregate per-pair focal estimates into one number and an honest variance.
///
/// Median rather than mean: a single pair whose homography latched onto a
/// moving object or a repeated texture produces a focal that is wrong by
/// hundreds of percent, and the mean would carry it straight through.
#[must_use]
pub fn aggregate_focals(samples: &[Scalar]) -> Option<FocalAggregate> {
    let usable: Vec<(usize, Scalar)> = samples
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite() && **v > 0.0)
        .map(|(i, v)| (i, *v))
        .collect();
    if usable.is_empty() {
        return None;
    }
    let values: Vec<Scalar> = usable.iter().map(|(_, v)| *v).collect();
    let centre = median(&values)?;

    // MAD first; it survives up to half the samples being wrong. Fall back to
    // the sample deviation only when the MAD collapses to zero, which happens
    // when more than half the samples are identical.
    let mut sigma = mad_sigma(&values).unwrap_or(0.0);
    if !(sigma > 0.0) && values.len() > 1 {
        let mean = values.iter().sum::<Scalar>() / values.len() as Scalar;
        let var = values.iter().map(|v| (v - mean).powi(2)).sum::<Scalar>()
            / (values.len() - 1) as Scalar;
        sigma = var.sqrt();
    }

    let (kept_idx, kept_vals): (Vec<usize>, Vec<Scalar>) = if sigma > 0.0 && values.len() >= 4 {
        usable
            .iter()
            .filter(|(_, v)| (v - centre).abs() <= OUTLIER_SIGMAS * sigma)
            .cloned()
            .unzip()
    } else {
        usable.iter().cloned().unzip()
    };
    if kept_vals.is_empty() {
        return None;
    }
    let focal_px = median(&kept_vals)?;

    // Re-measure the spread on the surviving set: the pre-rejection sigma is
    // inflated by exactly the samples we just removed.
    let mut kept_sigma = mad_sigma(&kept_vals).unwrap_or(0.0);
    if !(kept_sigma > 0.0) && kept_vals.len() > 1 {
        let mean = kept_vals.iter().sum::<Scalar>() / kept_vals.len() as Scalar;
        let var = kept_vals.iter().map(|v| (v - mean).powi(2)).sum::<Scalar>()
            / (kept_vals.len() - 1) as Scalar;
        kept_sigma = var.sqrt();
    }
    if !(kept_sigma > 0.0) {
        kept_sigma = SINGLE_SAMPLE_RELATIVE_SIGMA * focal_px;
    }

    let variance = MEDIAN_VARIANCE_FACTOR * kept_sigma * kept_sigma / kept_vals.len() as Scalar;
    Some(FocalAggregate {
        focal_px,
        variance,
        sigma: kept_sigma,
        kept: kept_idx,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use wslam_core::Vec3;

    /// The exact infinite homography for a centred principal point.
    fn infinite_homography(f: Scalar, r: &So3) -> Mat3 {
        let k = Mat3::new(f, 0.0, 0.0, 0.0, f, 0.0, 0.0, 0.0, 1.0);
        let k_inv = Mat3::new(1.0 / f, 0.0, 0.0, 0.0, 1.0 / f, 0.0, 0.0, 0.0, 1.0);
        k * r.matrix() * k_inv
    }

    #[test]
    fn focal_from_synthetic_infinite_homography_is_exact() {
        for &f in &[320.0, 517.3, 985.0, 2400.0] {
            for phi in [
                Vec3::new(0.0, 0.08, 0.0),
                Vec3::new(0.05, 0.0, 0.0),
                Vec3::new(0.03, -0.07, 0.11),
                Vec3::new(-0.2, 0.15, -0.05),
            ] {
                let r = So3::exp(&phi);
                let h = infinite_homography(f, &r);
                let got = focal_from_rotation_homography(&h, &r).unwrap();
                assert_relative_eq!(got, f, max_relative = 1e-12);
            }
        }
    }

    #[test]
    fn focal_is_invariant_to_homography_scale_and_sign() {
        let f = 985.0;
        let r = So3::exp(&Vec3::new(0.02, 0.06, -0.01));
        let h = infinite_homography(f, &r);
        for scale in [-3.7, -1.0, 1e-4, 1.0, 5.5e5] {
            let got = focal_from_rotation_homography(&(h * scale), &r).unwrap();
            assert_relative_eq!(got, f, max_relative = 1e-11);
        }
    }

    #[test]
    fn rotation_about_the_optical_axis_is_unobservable() {
        // Rolling the phone about the lens axis gives H = R regardless of f.
        // Any answer here would be pure noise amplification.
        let r = So3::exp(&Vec3::new(0.0, 0.0, 0.6));
        assert_relative_eq!(axis_tilt(&r), 0.0, epsilon = 1e-15);
        assert!(!rotation_is_observable(&r));
        let h = infinite_homography(985.0, &r);
        assert!(focal_from_rotation_homography(&h, &r).is_none());
    }

    #[test]
    fn near_identity_rotation_is_unobservable() {
        for angle in [0.0, 1e-6, 1e-3, 0.01] {
            let r = So3::exp(&Vec3::new(0.0, angle, 0.0));
            let h = infinite_homography(985.0, &r);
            assert!(
                focal_from_rotation_homography(&h, &r).is_none(),
                "angle {angle} rad must be rejected"
            );
        }
        // Just above the gate the answer comes back and it is right.
        let r = So3::exp(&Vec3::new(0.0, 0.03, 0.0));
        let h = infinite_homography(985.0, &r);
        assert_relative_eq!(
            focal_from_rotation_homography(&h, &r).unwrap(),
            985.0,
            max_relative = 1e-12
        );
    }

    #[test]
    fn axis_tilt_equals_sine_of_the_swept_angle() {
        // A pure yaw of theta sweeps the optical axis by exactly theta.
        for theta in [0.05, 0.3, 1.0] {
            let r = So3::exp(&Vec3::new(0.0, theta, 0.0));
            assert_relative_eq!(axis_tilt(&r), theta.sin(), epsilon = 1e-12);
        }
    }

    #[test]
    fn impossible_homography_returns_none_rather_than_a_number() {
        // Flip the third row's sign: f^2 comes out negative, which no pinhole
        // can produce.
        let f = 985.0;
        let r = So3::exp(&Vec3::new(0.0, 0.08, 0.0));
        let mut h = infinite_homography(f, &r);
        h[(2, 0)] = -h[(2, 0)];
        h[(2, 1)] = -h[(2, 1)];
        assert!(focal_from_rotation_homography(&h, &r).is_none());
    }

    #[test]
    fn aggregate_recovers_the_centre_and_rejects_gross_outliers() {
        let mut samples = vec![
            980.0, 990.0, 985.0, 992.0, 978.0, 988.0, 983.0, 995.0, 981.0, 987.0,
        ];
        let clean = aggregate_focals(&samples).unwrap();
        assert!((clean.focal_px - 985.0).abs() < 4.0);
        assert_eq!(clean.kept.len(), samples.len());

        samples.push(40_000.0);
        samples.push(12.0);
        let dirty = aggregate_focals(&samples).unwrap();
        assert!(
            (dirty.focal_px - clean.focal_px).abs() < 3.0,
            "median moved to {} from {}",
            dirty.focal_px,
            clean.focal_px
        );
        assert_eq!(dirty.kept.len(), 10, "both gross samples must be dropped");
    }

    #[test]
    fn aggregate_variance_shrinks_with_sample_count() {
        let mut rng = wslam_core::DeterministicRng::new("t", 4);
        let draw = |n: usize, rng: &mut wslam_core::DeterministicRng| {
            let s: Vec<Scalar> = (0..n).map(|_| rng.normal_with(985.0, 20.0)).collect();
            aggregate_focals(&s).unwrap().variance
        };
        let few = draw(6, &mut rng);
        let many = draw(200, &mut rng);
        assert!(many < few * 0.3, "variance {many} vs {few}");
    }

    #[test]
    fn aggregate_of_a_single_sample_is_not_certain() {
        let a = aggregate_focals(&[985.0]).unwrap();
        assert_relative_eq!(a.focal_px, 985.0);
        assert!(a.variance > 0.0, "a single pair cannot be certain");
        assert_relative_eq!(
            a.sigma,
            985.0 * SINGLE_SAMPLE_RELATIVE_SIGMA,
            epsilon = 1e-9
        );
    }

    #[test]
    fn aggregate_rejects_empty_and_non_physical_input() {
        assert!(aggregate_focals(&[]).is_none());
        assert!(aggregate_focals(&[-1.0, 0.0, Scalar::NAN]).is_none());
    }
}
