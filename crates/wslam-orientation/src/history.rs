//! Bounded attitude history, so `delta_rotation` can answer for times between
//! samples.

use std::collections::VecDeque;
use wslam_core::math::So3;
use wslam_core::time::Timestamp;

/// Attitudes retained for [`crate::OrientationFilter::delta_rotation`].
///
/// About 10 s at 100 Hz, 5 s at 200 Hz — comfortably longer than any
/// frame-to-frame interval L2 or L3 will ask about, and *bounded*. spec.md §9
/// lists unbounded memory as the thing that gets a phone tab killed; a growing
/// history here would be a slow leak for the life of the session, which is
/// exactly the shape of bug that only shows up on a user's device.
pub const HISTORY_CAPACITY: usize = 1024;

/// Ring buffer of `(timestamp, attitude)`, oldest first.
#[derive(Debug, Clone)]
pub(crate) struct AttitudeHistory {
    entries: VecDeque<(Timestamp, So3)>,
    capacity: usize,
}

impl AttitudeHistory {
    /// Empty history with a fixed capacity of at least one entry.
    pub(crate) fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        AttitudeHistory {
            entries: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Append, evicting the oldest entry at capacity.
    ///
    /// A timestamp at or before the newest entry replaces it rather than
    /// breaking the sort order the interpolation search depends on. The filter
    /// rejects non-increasing samples upstream, so this is a guard, not a path.
    pub(crate) fn push(&mut self, timestamp: Timestamp, attitude: So3) {
        if let Some(back) = self.entries.back_mut() {
            if timestamp <= back.0 {
                *back = (timestamp, attitude);
                return;
            }
        }
        if self.entries.len() == self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back((timestamp, attitude));
    }

    /// Attitude at `t`, geodesically interpolated between the bracketing
    /// samples. `None` when `t` falls outside the retained window.
    ///
    /// Interpolation is `r0 · exp(alpha · log(r0^-1 r1))` rather than
    /// `So3::slerp`, which delegates to `nalgebra` and panics on antipodal
    /// inputs. Consecutive samples are milliseconds apart so antipodal is
    /// unreachable in practice, but a panic in the pose path on a user's phone
    /// is not a risk worth carrying for two lines of code.
    pub(crate) fn at(&self, t: Timestamp) -> Option<So3> {
        let first = self.entries.front()?.0;
        let last = self.entries.back()?.0;
        if t < first || t > last {
            return None;
        }
        match self.entries.binary_search_by_key(&t, |e| e.0) {
            Ok(i) => Some(self.entries[i].1),
            Err(i) => {
                // The range check above puts the insertion point strictly
                // inside, so both neighbours exist.
                let (t0, r0) = self.entries[i - 1];
                let (t1, r1) = self.entries[i];
                let span = t1.since(t0);
                let alpha = if span <= 0.0 { 0.0 } else { t.since(t0) / span };
                Some(r0.plus(&(r1.minus(&r0) * alpha)))
            }
        }
    }

    /// Oldest and newest retained times.
    pub(crate) fn span(&self) -> Option<(Timestamp, Timestamp)> {
        Some((self.entries.front()?.0, self.entries.back()?.0))
    }

    /// Retained entry count. Only the capacity tests need it — the filter
    /// itself never asks, because the bound is the point and the count is not.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use wslam_core::math::Vec3;

    fn at(i: usize) -> Timestamp {
        Timestamp::from_seconds(i as f64 * 0.01)
    }
    fn rot(i: usize) -> So3 {
        So3::exp(&(Vec3::z() * (i as f64 * 0.01)))
    }

    #[test]
    fn capacity_is_a_hard_bound() {
        let mut h = AttitudeHistory::new(8);
        for i in 0..1000 {
            h.push(at(i), rot(i));
        }
        assert_eq!(h.len(), 8);
        let (lo, hi) = h.span().unwrap();
        assert_eq!(hi, at(999));
        assert_eq!(lo, at(992));
    }

    #[test]
    fn exact_sample_times_return_the_stored_attitude() {
        let mut h = AttitudeHistory::new(64);
        for i in 0..20 {
            h.push(at(i), rot(i));
        }
        for i in 0..20 {
            assert_relative_eq!(
                h.at(at(i)).unwrap().matrix(),
                rot(i).matrix(),
                epsilon = 1e-15
            );
        }
    }

    #[test]
    fn midpoint_interpolation_is_the_geodesic_midpoint() {
        let mut h = AttitudeHistory::new(4);
        h.push(Timestamp::from_seconds(0.0), So3::identity());
        h.push(Timestamp::from_seconds(1.0), So3::exp(&(Vec3::z() * 0.8)));
        let mid = h.at(Timestamp::from_seconds(0.25)).unwrap();
        assert_relative_eq!(mid.log(), Vec3::z() * 0.2, epsilon = 1e-12);
    }

    #[test]
    fn queries_outside_the_window_are_none_rather_than_extrapolated() {
        let mut h = AttitudeHistory::new(64);
        for i in 5..15 {
            h.push(at(i), rot(i));
        }
        assert!(h.at(at(4)).is_none());
        assert!(h.at(at(15)).is_none());
        assert!(h.at(at(5)).is_some());
        assert!(h.at(at(14)).is_some());
    }

    #[test]
    fn empty_history_answers_nothing() {
        let h = AttitudeHistory::new(16);
        assert!(h.at(Timestamp::ZERO).is_none());
        assert!(h.span().is_none());
        assert_eq!(h.len(), 0);
    }

    #[test]
    fn a_non_increasing_timestamp_replaces_rather_than_corrupting_the_order() {
        let mut h = AttitudeHistory::new(16);
        h.push(at(0), rot(0));
        h.push(at(5), rot(5));
        h.push(at(3), rot(3));
        assert_eq!(h.len(), 2);
        assert_eq!(h.span().unwrap().1, at(3));
        // Still searchable, i.e. still sorted.
        assert!(h.at(at(2)).is_some());
    }
}
