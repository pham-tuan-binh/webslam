//! The sliding state window a [`ScaleSource`] estimates from.
//!
//! spec.md §4 L5 gives the interface as
//! `estimate(window: StateWindow) -> { scale, variance } | null`. This is that
//! window: a bounded history of up-to-scale poses plus the inertial samples
//! interleaved with them, which is exactly the input a closed-form
//! visual-inertial scale solve needs — and enough for the other four sources to
//! do their (much easier) jobs.
//!
//! [`ScaleSource`]: https://docs.rs/wslam-scale

use crate::imu::ImuSample;
use crate::math::{Scalar, Se3, Vec3};
use crate::time::Timestamp;
use std::collections::VecDeque;

/// One entry in the window: an up-to-scale pose at a known time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowSample {
    /// Capture time, unified timebase.
    pub timestamp: Timestamp,
    /// Up-to-scale `T_world_camera` as produced by L3. Multiplying its
    /// translation by the estimated scale yields metres.
    pub pose: Se3,
    /// Number of map landmarks that participated in this frame's pose solve.
    /// A scale source may reasonably refuse to estimate from poorly constrained
    /// frames.
    pub landmark_count: usize,
}

/// Bounded history of up-to-scale poses and inertial samples.
///
/// Fixed capacity: a scale source must never be able to make the frontend's
/// memory grow without bound (spec.md §6 L4, "Map memory growth vs session
/// duration ... a phone tab will be killed if this is unbounded" — the same
/// discipline applies here).
#[derive(Debug, Clone)]
pub struct StateWindow {
    poses: VecDeque<WindowSample>,
    imu: VecDeque<ImuSample>,
    capacity_poses: usize,
    capacity_imu: usize,
}

impl StateWindow {
    /// A window holding at most `capacity_poses` poses and `capacity_imu`
    /// inertial samples.
    #[must_use]
    pub fn new(capacity_poses: usize, capacity_imu: usize) -> Self {
        StateWindow {
            poses: VecDeque::with_capacity(capacity_poses),
            imu: VecDeque::with_capacity(capacity_imu),
            capacity_poses: capacity_poses.max(1),
            capacity_imu: capacity_imu.max(1),
        }
    }

    /// Default sizing: ~4 s of pose history at 30 Hz and ~4 s of IMU at 100 Hz,
    /// which comfortably spans the 2 s Campos initialisation window
    /// (spec.md §5) without holding a whole session.
    #[must_use]
    pub fn with_default_capacity() -> Self {
        Self::new(120, 400)
    }

    /// Append a pose sample, evicting the oldest if at capacity.
    pub fn push_pose(&mut self, sample: WindowSample) {
        if self.poses.len() == self.capacity_poses {
            self.poses.pop_front();
        }
        self.poses.push_back(sample);
    }

    /// Append an inertial sample, evicting the oldest if at capacity.
    pub fn push_imu(&mut self, sample: ImuSample) {
        if self.imu.len() == self.capacity_imu {
            self.imu.pop_front();
        }
        self.imu.push_back(sample);
    }

    /// Pose samples, oldest first.
    pub fn poses(&self) -> impl Iterator<Item = &WindowSample> + '_ {
        self.poses.iter()
    }

    /// Inertial samples, oldest first.
    pub fn imu(&self) -> impl Iterator<Item = &ImuSample> + '_ {
        self.imu.iter()
    }

    /// Number of pose samples held.
    #[must_use]
    pub fn pose_count(&self) -> usize {
        self.poses.len()
    }

    /// Number of inertial samples held.
    #[must_use]
    pub fn imu_count(&self) -> usize {
        self.imu.len()
    }

    /// Most recent pose sample.
    #[must_use]
    pub fn latest_pose(&self) -> Option<&WindowSample> {
        self.poses.back()
    }

    /// Oldest pose sample.
    #[must_use]
    pub fn oldest_pose(&self) -> Option<&WindowSample> {
        self.poses.front()
    }

    /// Time span covered by the pose history, in seconds.
    #[must_use]
    pub fn span_seconds(&self) -> Scalar {
        match (self.poses.front(), self.poses.back()) {
            (Some(a), Some(b)) => b.timestamp.since(a.timestamp),
            _ => 0.0,
        }
    }

    /// Total path length of the up-to-scale trajectory, in window units.
    /// A scale source divides a metric length by this to recover a multiplier.
    #[must_use]
    pub fn path_length(&self) -> Scalar {
        self.poses
            .iter()
            .zip(self.poses.iter().skip(1))
            .map(|(a, b)| (b.pose.translation() - a.pose.translation()).norm())
            .sum()
    }

    /// Straight-line displacement between the oldest and newest pose.
    #[must_use]
    pub fn displacement(&self) -> Vec3 {
        match (self.poses.front(), self.poses.back()) {
            (Some(a), Some(b)) => b.pose.translation() - a.pose.translation(),
            _ => Vec3::zeros(),
        }
    }

    /// Mean absolute translational excitation over the inertial history.
    ///
    /// spec.md §6 L5: *"Report the excitation dependence explicitly ... the
    /// theory says accuracy collapses as translational acceleration vanishes."*
    /// A scale source that needs excitation should consult this and decline
    /// rather than return a confident wrong answer — which is also the
    /// "static hold ... should be *detected* rather than silently wrong"
    /// requirement from the Tier-3 trajectory set.
    #[must_use]
    pub fn mean_excitation(&self) -> Scalar {
        if self.imu.is_empty() {
            return 0.0;
        }
        let g = crate::imu::GRAVITY;
        self.imu.iter().map(|s| s.excitation(g)).sum::<Scalar>() / self.imu.len() as Scalar
    }

    /// Peak angular rate over the inertial history, rad/s. Used to detect the
    /// pure-rotation degenerate case for inertial scale.
    #[must_use]
    pub fn peak_angular_rate(&self) -> Scalar {
        self.imu
            .iter()
            .map(|s| s.gyro.norm())
            .fold(0.0, Scalar::max)
    }

    /// Inertial samples strictly inside `[from, to]`.
    pub fn imu_between(
        &self,
        from: Timestamp,
        to: Timestamp,
    ) -> impl Iterator<Item = &ImuSample> + '_ {
        self.imu
            .iter()
            .filter(move |s| s.timestamp >= from && s.timestamp <= to)
    }

    /// Drop everything. Called on re-initialisation, so a stale window cannot
    /// anchor a fresh session.
    pub fn clear(&mut self) {
        self.poses.clear();
        self.imu.clear();
    }
}

impl Default for StateWindow {
    fn default() -> Self {
        Self::with_default_capacity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imu::GRAVITY;
    use crate::math::So3;
    use approx::assert_relative_eq;

    fn pose_at(t: f64, x: f64) -> WindowSample {
        WindowSample {
            timestamp: Timestamp::from_seconds(t),
            pose: Se3::new(So3::identity(), Vec3::new(x, 0.0, 0.0)),
            landmark_count: 50,
        }
    }

    #[test]
    fn window_evicts_at_capacity() {
        let mut w = StateWindow::new(3, 3);
        for i in 0..5 {
            w.push_pose(pose_at(i as f64, i as f64));
        }
        assert_eq!(w.pose_count(), 3);
        assert_relative_eq!(w.oldest_pose().unwrap().pose.translation().x, 2.0);
        assert_relative_eq!(w.latest_pose().unwrap().pose.translation().x, 4.0);
    }

    #[test]
    fn path_length_and_displacement_differ_on_a_there_and_back() {
        let mut w = StateWindow::new(10, 10);
        w.push_pose(pose_at(0.0, 0.0));
        w.push_pose(pose_at(1.0, 1.0));
        w.push_pose(pose_at(2.0, 0.0));
        assert_relative_eq!(w.path_length(), 2.0, epsilon = 1e-12);
        assert_relative_eq!(w.displacement().norm(), 0.0, epsilon = 1e-12);
        assert_relative_eq!(w.span_seconds(), 2.0, epsilon = 1e-12);
    }

    #[test]
    fn empty_window_reports_zeroes_not_nan() {
        let w = StateWindow::new(4, 4);
        assert_eq!(w.span_seconds(), 0.0);
        assert_eq!(w.path_length(), 0.0);
        assert_eq!(w.mean_excitation(), 0.0);
        assert_eq!(w.peak_angular_rate(), 0.0);
        assert!(w.latest_pose().is_none());
    }

    #[test]
    fn static_hold_reports_near_zero_excitation() {
        // The degenerate case for inertial scale, which must be detectable.
        let mut w = StateWindow::new(4, 100);
        for i in 0..50 {
            w.push_imu(ImuSample::new(
                Timestamp::from_seconds(i as f64 * 0.01),
                Vec3::zeros(),
                Vec3::new(0.0, 0.0, GRAVITY),
            ));
        }
        assert!(w.mean_excitation() < 1e-9);
    }

    #[test]
    fn shaking_reports_high_excitation() {
        let mut w = StateWindow::new(4, 100);
        for i in 0..50 {
            let a = (i as f64 * 0.6).sin() * 3.0;
            w.push_imu(ImuSample::new(
                Timestamp::from_seconds(i as f64 * 0.01),
                Vec3::zeros(),
                Vec3::new(a, 0.0, GRAVITY),
            ));
        }
        assert!(w.mean_excitation() > 0.1, "{}", w.mean_excitation());
    }

    #[test]
    fn imu_between_is_inclusive() {
        let mut w = StateWindow::new(4, 100);
        for i in 0..10 {
            w.push_imu(ImuSample::new(
                Timestamp::from_seconds(i as f64),
                Vec3::zeros(),
                Vec3::zeros(),
            ));
        }
        let n = w
            .imu_between(Timestamp::from_seconds(2.0), Timestamp::from_seconds(5.0))
            .count();
        assert_eq!(n, 4);
    }
}
