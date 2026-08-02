//! Trajectory metrics.
//!
//! spec.md §6 is specific about which number each layer is graded on, and the
//! two that matter here are:
//!
//! - **L3**: *"ATE after Sim(3) alignment (scale-free — L3 does not claim
//!   scale)"*. Aligning with scale free is not a convenience; comparing an
//!   up-to-scale trajectory to metric ground truth any other way measures the
//!   arbitrary unit choice rather than the tracker.
//! - **L6**: NEES against chi-squared bounds and empirical coverage. That lives
//!   in `wslam_core::covariance` so the harness and the library cannot drift
//!   apart in how they compute it.

use wslam_core::math::umeyama;
use wslam_core::{Scalar, Se3, Timestamp, Vec3};

/// One pose in a trajectory.
#[derive(Debug, Clone, Copy)]
pub struct Stamped {
    /// Capture time in the unified timebase.
    pub timestamp: Timestamp,
    /// `T_world_camera`.
    pub pose: Se3,
    /// Coordinate-frame epoch. Poses from different epochs have unrelated
    /// origins and scales and must never share an alignment.
    pub epoch: u32,
}

/// Absolute trajectory error after alignment.
#[derive(Debug, Clone, Copy)]
pub struct AteReport {
    /// Root-mean-square position error, in ground-truth units.
    pub rmse: Scalar,
    /// Median position error — more robust than the RMSE to a single spike,
    /// and reported alongside so a bimodal failure is visible.
    pub median: Scalar,
    /// Worst position error.
    pub max: Scalar,
    /// Pose pairs that were matched and compared.
    pub pairs: usize,
    /// Scale recovered by the alignment.
    ///
    /// For an up-to-scale estimate this is the arbitrary unit conversion. For a
    /// metric one it should be 1.0, and its deviation **is** the scale error
    /// spec.md §6 L5 asks for.
    pub scale: Scalar,
    /// Fraction of emitted poses with no ground truth inside the match
    /// tolerance. Distinct from the tracker's own loss rate, which
    /// `SequenceReport::lost_fraction` reports.
    pub unmatched_fraction: Scalar,
}

impl std::fmt::Display for AteReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ATE rmse {:.4} median {:.4} max {:.4} over {} pairs (scale {:.4}, {:.1}% unmatched)",
            self.rmse,
            self.median,
            self.max,
            self.pairs,
            self.scale,
            100.0 * self.unmatched_fraction
        )
    }
}

/// Match estimated poses to ground truth by nearest timestamp, **one to one**.
///
/// `tolerance` is the largest acceptable time difference, in seconds.
///
/// Iterates over *estimates* and finds the nearest ground-truth pose for each,
/// not the other way round. Ground truth is usually far denser than the
/// estimate — EuRoC publishes 200 Hz mocap against a 20 Hz camera — so walking
/// ground truth matches ten of its poses to the same estimate, inflating the
/// pair count tenfold and weighting the ATE by however dense the truth happened
/// to be. Measured on MH_01_easy: 145 estimates produced 1113 "pairs".
///
/// Returns `(estimates, truth, unmatched_estimates)`. An estimate with no truth
/// inside the tolerance is counted, not silently matched to a distant pose — a
/// 20 ms mismatch during fast motion is a position error that has nothing to do
/// with the estimator.
#[must_use]
pub fn associate(
    estimated: &[Stamped],
    truth: &[Stamped],
    tolerance: Scalar,
) -> (Vec<Se3>, Vec<Se3>, usize) {
    let mut est_out = Vec::new();
    let mut truth_out = Vec::new();
    let mut unmatched = 0usize;
    if truth.is_empty() {
        return (est_out, truth_out, estimated.len());
    }

    // Both streams are time-ordered, so one advancing cursor is enough.
    let mut cursor = 0usize;
    for sample in estimated {
        while cursor + 1 < truth.len()
            && (truth[cursor + 1].timestamp.since(sample.timestamp)).abs()
                < (truth[cursor].timestamp.since(sample.timestamp)).abs()
        {
            cursor += 1;
        }
        if truth[cursor].timestamp.since(sample.timestamp).abs() <= tolerance {
            est_out.push(sample.pose);
            truth_out.push(truth[cursor].pose);
        } else {
            unmatched += 1;
        }
    }
    (est_out, truth_out, unmatched)
}

/// Split a trajectory at coordinate-frame boundaries.
///
/// Poses from different epochs have unrelated origins and scales, so any metric
/// that fits a single transform across them is measuring the seams. ORB-SLAM3
/// does the same thing in `SaveTrajectoryEuRoC`, evaluating only the largest
/// map rather than concatenating them.
#[must_use]
pub fn split_by_epoch(estimated: &[Stamped]) -> Vec<Vec<Stamped>> {
    let mut out: Vec<Vec<Stamped>> = Vec::new();
    for s in estimated {
        match out.last_mut() {
            Some(seg) if seg.last().is_some_and(|p| p.epoch == s.epoch) => seg.push(*s),
            _ => out.push(vec![*s]),
        }
    }
    out
}

/// ATE over the **largest** coordinate-frame segment, plus its coverage.
///
/// Reporting one number for a spliced trajectory measures the discontinuities
/// rather than the tracking: on EuRoC MH_01 that read 3.1 m where the segments
/// themselves sit near 0.06 m. Coverage is returned alongside and must be
/// reported with it — a system that keeps one good 200-frame segment out of
/// 3682 has not earned a good ATE, which is the same reason the baselines guard
/// `lost_fraction`.
#[must_use]
pub fn ate_largest_segment(
    estimated: &[Stamped],
    truth: &[Stamped],
    tolerance: Scalar,
    estimate_scale: bool,
) -> Option<(AteReport, usize, Scalar)> {
    let segments = split_by_epoch(estimated);
    let largest = segments.iter().max_by_key(|s| s.len())?;
    let coverage = largest.len() as Scalar / estimated.len().max(1) as Scalar;
    let report = ate(largest, truth, tolerance, estimate_scale)?;
    Some((report, segments.len(), coverage))
}

/// Absolute trajectory error.
///
/// `estimate_scale` selects Sim(3) alignment (scale free, for L3) versus SE(3)
/// alignment (for a trajectory that claims metres). Returns `None` when fewer
/// than three poses could be associated, because the alignment is undetermined
/// below that and a number computed from two points would be meaningless.
#[must_use]
pub fn ate(
    estimated: &[Stamped],
    truth: &[Stamped],
    tolerance: Scalar,
    estimate_scale: bool,
) -> Option<AteReport> {
    let (est, gt, unmatched) = associate(estimated, truth, tolerance);
    if est.len() < 3 {
        return None;
    }
    let src: Vec<Vec3> = est.iter().map(Se3::translation).collect();
    let dst: Vec<Vec3> = gt.iter().map(Se3::translation).collect();
    let alignment = umeyama(&src, &dst, estimate_scale)?;

    let mut errors: Vec<Scalar> = src
        .iter()
        .zip(dst.iter())
        .map(|(s, d)| (d - alignment.transform.act(s)).norm())
        .collect();

    let sum_sq: Scalar = errors.iter().map(|e| e * e).sum();
    let rmse = (sum_sq / errors.len() as Scalar).sqrt();
    errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    Some(AteReport {
        rmse,
        median: errors[errors.len() / 2],
        max: *errors.last().expect("non-empty"),
        pairs: errors.len(),
        scale: alignment.transform.scale(),
        unmatched_fraction: unmatched as Scalar / estimated.len().max(1) as Scalar,
    })
}

/// Relative pose error over a fixed time delta, in **ground-truth units**.
///
/// Complements ATE: a trajectory can have small RPE and large ATE (slow drift)
/// or the reverse (a single jump). Reporting only one hides half the failures.
///
/// The estimate is rescaled by the Sim(3) alignment scale before differencing.
/// Without that, an up-to-scale trajectory reports its RPE in arbitrary units —
/// measured on MH_01_easy the raw number was 4.25 "m" for a trajectory running
/// 23x larger than truth, i.e. really 0.18 m. A number whose unit depends on
/// where the bootstrap happened to normalise its baseline is not a metric.
#[must_use]
pub fn rpe(
    estimated: &[Stamped],
    truth: &[Stamped],
    delta_seconds: Scalar,
    tolerance: Scalar,
) -> Option<(Scalar, Scalar, Scalar)> {
    let (est, gt, _) = associate(estimated, truth, tolerance);
    if est.len() < 3 {
        return None;
    }
    // Put the estimate into ground-truth units first; see the doc comment.
    let scale = {
        let src: Vec<Vec3> = est.iter().map(Se3::translation).collect();
        let dst: Vec<Vec3> = gt.iter().map(Se3::translation).collect();
        umeyama(&src, &dst, true)?.transform.scale()
    };
    let est: Vec<Se3> = est.iter().map(|p| p.scaled(scale)).collect();

    // The stride indexes the *associated* arrays, which are one entry per
    // estimate — so it must come from the estimate's own spacing, not from the
    // truth's. Taking it from the truth was a factor-of-ten error on EuRoC,
    // where 200 Hz mocap meets a 20 Hz camera: a requested 1 s window measured
    // 10 s of drift and reported it as 1 s.
    let spacing = {
        let mut gaps: Vec<Scalar> = estimated
            .windows(2)
            .map(|w| w[1].timestamp.since(w[0].timestamp))
            .filter(|g| *g > 0.0)
            .collect();
        if gaps.is_empty() {
            return None;
        }
        gaps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        gaps[gaps.len() / 2]
    };
    let stride = ((delta_seconds / spacing).round() as usize).max(1);
    if est.len() <= stride {
        return None;
    }
    let effective_delta = stride as Scalar * spacing;

    let mut translation = Vec::new();
    let mut rotation = Vec::new();
    for i in 0..(est.len() - stride) {
        let de = est[i].inverse().compose(&est[i + stride]);
        let dg = gt[i].inverse().compose(&gt[i + stride]);
        let error = dg.inverse().compose(&de);
        translation.push(error.translation().norm());
        rotation.push(error.rotation().angle());
    }
    let rms = |v: &[Scalar]| (v.iter().map(|x| x * x).sum::<Scalar>() / v.len() as Scalar).sqrt();
    Some((rms(&translation), rms(&rotation), effective_delta))
}

/// Scale error as a percentage, against metric ground truth.
///
/// **The headline metric.** spec.md §6 L5: *"scale error % as a function of
/// time-since-init. Directly reproduce the Campos curve: report our 2 s and
/// 10 s numbers against their 5% / 1%."*
///
/// Computed from the ratio of estimated to true path length, which is what a
/// consumer actually cares about, rather than from the Sim(3) alignment scale —
/// the two agree for a rigid trajectory but the path-length ratio degrades
/// gracefully when the trajectory is partly lost.
#[must_use]
pub fn scale_error_percent(
    estimated: &[Stamped],
    truth: &[Stamped],
    tolerance: Scalar,
) -> Option<Scalar> {
    let (est, gt, _) = associate(estimated, truth, tolerance);
    if est.len() < 3 {
        return None;
    }
    let path = |poses: &[Se3]| -> Scalar {
        poses
            .windows(2)
            .map(|w| (w[1].translation() - w[0].translation()).norm())
            .sum()
    };
    let (est_len, gt_len) = (path(&est), path(&gt));
    if gt_len < 1e-9 {
        // A stationary ground truth carries no scale information. Returning
        // zero error here would be a free pass on exactly the degenerate case
        // spec.md §6 Tier 3 says must be detected.
        return None;
    }
    Some(100.0 * (est_len / gt_len - 1.0).abs())
}

/// Campos-curve samples: scale error at fixed times since initialisation.
///
/// spec.md §6 L5 asks for the 2 s and 10 s numbers specifically, against
/// Campos et al.'s 5% and 1%.
///
/// **Only meaningful for a metric session.** Called on an up-to-scale trajectory
/// it reports the arbitrary bootstrap baseline as a scale error — measured 1251%
/// at 2 s on MH_01_easy, a number that says nothing about anything. The caller
/// must gate on `ScaleKind::is_metric`; `SequenceReport::build` does.
#[must_use]
pub fn campos_curve(
    estimated: &[Stamped],
    truth: &[Stamped],
    tolerance: Scalar,
    checkpoints: &[Scalar],
) -> Vec<(Scalar, Option<Scalar>)> {
    let Some(t0) = estimated.first().map(|s| s.timestamp) else {
        return checkpoints.iter().map(|t| (*t, None)).collect();
    };
    checkpoints
        .iter()
        .map(|&seconds| {
            let cutoff = t0.offset_seconds(seconds);
            let est: Vec<Stamped> = estimated
                .iter()
                .copied()
                .filter(|s| s.timestamp <= cutoff)
                .collect();
            (seconds, scale_error_percent(&est, truth, tolerance))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use wslam_core::So3;

    fn straight_line(n: usize, step: Scalar, hz: Scalar) -> Vec<Stamped> {
        (0..n)
            .map(|i| Stamped {
                timestamp: Timestamp::from_seconds(i as Scalar / hz),
                pose: Se3::new(So3::identity(), Vec3::new(i as Scalar * step, 0.0, 0.0)),
                epoch: 0,
            })
            .collect()
    }

    #[test]
    fn a_perfect_estimate_has_zero_error() {
        let truth = straight_line(50, 0.05, 30.0);
        let report = ate(&truth, &truth, 0.02, true).expect("report");
        assert!(report.rmse < 1e-12, "{report}");
        assert_relative_eq!(report.scale, 1.0, epsilon = 1e-9);
        assert_eq!(report.pairs, 50);
        assert_eq!(report.unmatched_fraction, 0.0);
    }

    #[test]
    fn sim3_alignment_absorbs_an_arbitrary_scale() {
        // This is the property that makes ATE meaningful for L3, which makes no
        // metric claim. Without it the number would measure the unit choice.
        let truth = straight_line(50, 0.05, 30.0);
        let estimate: Vec<Stamped> = truth
            .iter()
            .map(|s| Stamped {
                timestamp: s.timestamp,
                pose: s.pose.scaled(7.3),
                epoch: 0,
            })
            .collect();
        let report = ate(&estimate, &truth, 0.02, true).expect("report");
        assert!(report.rmse < 1e-9, "{report}");
        assert_relative_eq!(report.scale, 1.0 / 7.3, epsilon = 1e-9);
    }

    #[test]
    fn se3_alignment_does_not_absorb_scale() {
        let truth = straight_line(50, 0.05, 30.0);
        let estimate: Vec<Stamped> = truth
            .iter()
            .map(|s| Stamped {
                timestamp: s.timestamp,
                pose: s.pose.scaled(2.0),
                epoch: 0,
            })
            .collect();
        let report = ate(&estimate, &truth, 0.02, false).expect("report");
        assert!(
            report.rmse > 0.1,
            "scale-locked ATE must see the error: {report}"
        );
        assert_relative_eq!(report.scale, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn a_known_offset_produces_the_expected_rmse() {
        // Aligning absorbs a constant offset entirely, so perturb only half the
        // trajectory to leave something the alignment cannot remove.
        let truth = straight_line(40, 0.05, 30.0);
        let estimate: Vec<Stamped> = truth
            .iter()
            .enumerate()
            .map(|(i, s)| Stamped {
                timestamp: s.timestamp,
                pose: if i >= 20 {
                    Se3::new(
                        s.pose.rotation(),
                        s.pose.translation() + Vec3::new(0.0, 0.1, 0.0),
                    )
                } else {
                    s.pose
                },
                epoch: 0,
            })
            .collect();
        let report = ate(&estimate, &truth, 0.02, true).expect("report");
        assert!(report.rmse > 0.01, "{report}");
        assert!(report.max >= report.median);
    }

    #[test]
    fn one_estimate_matches_exactly_one_ground_truth_pose() {
        // The bug this pins: ground truth is denser than the estimate (EuRoC
        // publishes 200 Hz mocap against a 20 Hz camera), so walking ground
        // truth matched ~8 of its poses to each estimate and reported 1113
        // "pairs" for 145 estimates. The ATE was then weighted by however dense
        // the truth happened to be.
        let truth = straight_line(200, 0.01, 200.0); // 200 Hz
        let estimate: Vec<Stamped> = truth.iter().step_by(10).copied().collect(); // 20 Hz
        let (est, gt, unmatched) = associate(&estimate, &truth, 0.02);
        assert_eq!(est.len(), estimate.len());
        assert_eq!(gt.len(), estimate.len());
        assert_eq!(unmatched, 0);
    }

    #[test]
    fn estimates_beyond_the_ground_truth_are_counted_not_matched() {
        // A tracker that keeps emitting after the mocap stops must not have
        // those poses silently paired with the last known truth.
        let truth = straight_line(20, 0.05, 30.0);
        let mut estimate = truth.clone();
        for i in 0..20 {
            estimate.push(Stamped {
                timestamp: Timestamp::from_seconds(100.0 + i as Scalar),
                pose: Se3::identity(),
                epoch: 0,
            });
        }
        let report = ate(&estimate, &truth, 0.02, true).expect("report");
        assert!(
            (report.unmatched_fraction - 0.5).abs() < 1e-9,
            "expected half the estimates unmatched, got {:.3}",
            report.unmatched_fraction
        );
        assert_eq!(report.pairs, 20);
    }

    #[test]
    fn association_refuses_a_distant_match() {
        let truth = straight_line(10, 0.05, 30.0);
        let shifted: Vec<Stamped> = truth
            .iter()
            .map(|s| Stamped {
                timestamp: s.timestamp.offset_seconds(5.0),
                pose: s.pose,
                epoch: 0,
            })
            .collect();
        let (est, gt, missing) = associate(&shifted, &truth, 0.02);
        assert!(est.is_empty() && gt.is_empty());
        assert_eq!(missing, 10);
    }

    #[test]
    fn too_few_poses_gives_no_report_rather_than_a_meaningless_one() {
        let truth = straight_line(2, 0.05, 30.0);
        assert!(ate(&truth, &truth, 0.02, true).is_none());
        assert!(ate(&[], &truth, 0.02, true).is_none());
    }

    #[test]
    fn scale_error_recovers_a_known_ratio() {
        let truth = straight_line(50, 0.05, 30.0);
        let estimate: Vec<Stamped> = truth
            .iter()
            .map(|s| Stamped {
                timestamp: s.timestamp,
                pose: s.pose.scaled(1.05),
                epoch: 0,
            })
            .collect();
        let error = scale_error_percent(&estimate, &truth, 0.02).expect("error");
        assert_relative_eq!(error, 5.0, epsilon = 1e-9);
    }

    #[test]
    fn a_stationary_ground_truth_yields_no_scale_number() {
        // The degenerate case. Reporting 0% error here would be a free pass on
        // exactly the trajectory spec.md §6 Tier 3 says must be detected.
        let truth: Vec<Stamped> = (0..30)
            .map(|i| Stamped {
                timestamp: Timestamp::from_seconds(i as Scalar / 30.0),
                pose: Se3::identity(),
                epoch: 0,
            })
            .collect();
        assert!(scale_error_percent(&truth, &truth, 0.02).is_none());
    }

    #[test]
    fn the_campos_curve_reports_a_value_per_checkpoint() {
        let truth = straight_line(600, 0.02, 30.0);
        let estimate: Vec<Stamped> = truth
            .iter()
            .map(|s| Stamped {
                timestamp: s.timestamp,
                pose: s.pose.scaled(1.02),
                epoch: 0,
            })
            .collect();
        let curve = campos_curve(&estimate, &truth, 0.02, &[2.0, 10.0]);
        assert_eq!(curve.len(), 2);
        for (at, error) in curve {
            let error = error.unwrap_or_else(|| panic!("no value at {at}s"));
            assert_relative_eq!(error, 2.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn rpe_is_zero_for_a_perfect_estimate() {
        let truth = straight_line(120, 0.05, 30.0);
        let (t, r, delta) = rpe(&truth, &truth, 1.0, 0.02).expect("rpe");
        assert!(t < 1e-12 && r < 1e-12);
        // The stride is an integer number of samples, so the effective window
        // lands within one sample period of the request — not exactly on it,
        // because 1/30 s is not representable in integer nanoseconds.
        assert!(
            (delta - 1.0).abs() < 1.0 / 30.0,
            "effective delta {delta} is more than one sample from the requested 1.0 s"
        );
    }

    #[test]
    fn rpe_sees_a_local_jump_that_ate_alignment_would_dilute() {
        let truth = straight_line(120, 0.05, 30.0);
        let mut estimate = truth.clone();
        for s in estimate.iter_mut().skip(60) {
            s.pose = Se3::new(
                s.pose.rotation(),
                s.pose.translation() + Vec3::new(0.0, 0.5, 0.0),
            );
        }
        let (t, _, _) = rpe(&estimate, &truth, 1.0, 0.02).expect("rpe");
        assert!(t > 0.01, "RPE should see the step: {t}");
    }
}
