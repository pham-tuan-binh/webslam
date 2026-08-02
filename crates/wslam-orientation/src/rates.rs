//! The stillness detector's input: a short rolling window of measured rates.
//!
//! ## Why a window and not the newest sample
//!
//! The zero-angular-rate update needs to know whether the device is turning.
//! The obvious test — "is `|omega_measured|` below the threshold?" — is wrong,
//! and wrong in a way that quietly destroys the very state it is trying to
//! observe.
//!
//! A phone's gyro is a *noise density*: `OrientationConfig::gyro_noise` is
//! rad/s/sqrt(Hz), so a single sample at 100 Hz carries a standard deviation of
//! `gyro_noise / sqrt(dt)` — 0.02 rad/s for the defaults. That is the same size
//! as the whole `static_threshold`. Thresholding a measurement whose noise
//! exceeds the threshold does not select still samples; it selects samples whose
//! *noise* happened to cancel the bias. Feeding those to the filter as an
//! unbiased observation of the bias is textbook selection bias, and it shrinks
//! the estimate toward zero: measured over 30 000 synthetic samples with a true
//! 0.005 rad/s bias, the accepted mean is 0.0008 rad/s — a sixth of the truth.
//!
//! Averaging first fixes it. The mean over `T` seconds has standard deviation
//! `gyro_noise / sqrt(T)`, so half a second of samples brings it to 0.003 rad/s,
//! comfortably under the threshold, and the same experiment recovers 0.00505.
//! The window is also consulted *before* the current sample joins it, so the
//! sample used as the measurement is statistically independent of the decision
//! to use it.
//!
//! ## What it still cannot do
//!
//! A device turning at a genuinely constant rate below the threshold is
//! indistinguishable from a biased one at rest, from the gyro alone and for any
//! detector. That ambiguity is priced into the update as measurement noise
//! (`static_threshold / 3`), not detected away.

use std::collections::VecDeque;

use wslam_core::math::{Scalar, Vec3};
use wslam_core::time::Timestamp;

/// Seconds of angular rate averaged by the stillness detector.
///
/// Long enough that the mean's noise is well under `static_threshold` (0.003
/// against 0.02 rad/s at the default noise density), short enough that half a
/// second of stillness inside ordinary handling still earns a bias update.
pub(crate) const STATIC_WINDOW_SECONDS: Scalar = 0.5;

/// Samples below which a mean is not worth forming, however long the span.
const MIN_SAMPLES: usize = 8;

/// Rolling window of measured angular rates over the last
/// [`STATIC_WINDOW_SECONDS`].
///
/// Bounded by time rather than by count, so it costs the same at 30 Hz and at
/// 200 Hz, and a stalled event stream empties it instead of averaging across
/// the gap.
#[derive(Debug, Clone)]
pub(crate) struct RateWindow {
    entries: VecDeque<(Timestamp, Vec3)>,
    sum: Vec3,
    span_seconds: Scalar,
}

impl RateWindow {
    /// Empty window covering `span_seconds`.
    pub(crate) fn new(span_seconds: Scalar) -> Self {
        RateWindow {
            entries: VecDeque::new(),
            sum: Vec3::zeros(),
            span_seconds,
        }
    }

    /// Append a measured rate, evicting everything older than the span.
    ///
    /// The running sum is rebuilt from the retained entries whenever it could
    /// have drifted: summing incrementally over a 15-minute session
    /// (spec.md §6, thermal soak) accumulates rounding that a subtraction never
    /// takes back, and the window is short enough that an exact resum is cheap.
    pub(crate) fn push(&mut self, timestamp: Timestamp, gyro: Vec3) {
        if let Some(&(last, _)) = self.entries.back() {
            if timestamp <= last {
                // The filter refuses non-increasing samples upstream; treat one
                // that arrives anyway as a restart rather than corrupting the
                // ordering the eviction loop relies on.
                self.entries.clear();
            }
        }
        self.entries.push_back((timestamp, gyro));
        while let Some(&(front, _)) = self.entries.front() {
            if self.entries.len() > 1 && timestamp.since(front) > self.span_seconds {
                self.entries.pop_front();
            } else {
                break;
            }
        }
        self.sum = self.entries.iter().map(|e| e.1).sum();
    }

    /// Mean measured rate and the duration it averages, as observed from `now`.
    ///
    /// `None` when the window is too short to be a useful statistic, or when it
    /// is stale — a gap longer than the span means the samples in hand describe
    /// a different moment than `now` does.
    pub(crate) fn mean(&self, now: Timestamp) -> Option<(Vec3, Scalar)> {
        let (first, _) = *self.entries.front()?;
        let (last, _) = *self.entries.back()?;
        if self.entries.len() < MIN_SAMPLES || now.since(last) > self.span_seconds {
            return None;
        }
        let span = last.since(first);
        if span < 0.5 * self.span_seconds {
            return None;
        }
        // One sample covers one interval, so n samples average n intervals, not
        // the n-1 the endpoints span. The correction matters at low rates.
        let n = self.entries.len() as Scalar;
        let covered = span * n / (n - 1.0);
        Some((self.sum / n, covered))
    }

    /// Retained sample count.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn at(i: usize) -> Timestamp {
        Timestamp::from_seconds(i as Scalar * 0.01)
    }

    #[test]
    fn the_window_is_bounded_by_time_not_by_count() {
        let mut w = RateWindow::new(STATIC_WINDOW_SECONDS);
        for i in 0..1000 {
            w.push(at(i), Vec3::x());
        }
        // 0.5 s at 100 Hz, plus the boundary sample that is exactly 0.5 s old.
        assert_eq!(w.len(), 51);
    }

    #[test]
    fn the_mean_is_the_mean_and_the_duration_counts_intervals() {
        let mut w = RateWindow::new(STATIC_WINDOW_SECONDS);
        for i in 0..=50 {
            let v = if i % 2 == 0 { 0.04 } else { -0.02 };
            w.push(at(i), Vec3::new(v, 0.0, 0.0));
        }
        let (mean, covered) = w.mean(at(50)).unwrap();
        // 26 samples at +0.04, 25 at -0.02.
        assert_relative_eq!(mean.x, (26.0 * 0.04 - 25.0 * 0.02) / 51.0, epsilon = 1e-15);
        assert_relative_eq!(mean.y, 0.0, epsilon = 1e-15);
        // 51 samples at 10 ms is 0.51 s of averaging, not the 0.50 s spanned.
        assert_relative_eq!(covered, 0.51, epsilon = 1e-12);
    }

    #[test]
    fn a_short_or_sparse_window_declines_to_answer() {
        let mut w = RateWindow::new(STATIC_WINDOW_SECONDS);
        assert!(w.mean(at(0)).is_none());
        for i in 0..MIN_SAMPLES {
            w.push(at(i), Vec3::zeros());
        }
        // Enough samples, but they only span 70 ms.
        assert!(w.mean(at(MIN_SAMPLES - 1)).is_none());

        // Long enough span, too few samples to average anything.
        let mut sparse = RateWindow::new(STATIC_WINDOW_SECONDS);
        sparse.push(Timestamp::from_seconds(0.0), Vec3::zeros());
        sparse.push(Timestamp::from_seconds(0.4), Vec3::zeros());
        assert!(sparse.mean(Timestamp::from_seconds(0.4)).is_none());
    }

    #[test]
    fn a_stalled_stream_makes_the_window_stale_rather_than_wrong() {
        let mut w = RateWindow::new(STATIC_WINDOW_SECONDS);
        for i in 0..=50 {
            w.push(at(i), Vec3::zeros());
        }
        assert!(w.mean(at(50)).is_some());
        // Three seconds later the window describes a moment that has passed.
        assert!(w.mean(Timestamp::from_seconds(3.5)).is_none());
    }

    #[test]
    fn an_out_of_order_sample_restarts_the_window() {
        let mut w = RateWindow::new(STATIC_WINDOW_SECONDS);
        for i in 0..=50 {
            w.push(at(i), Vec3::zeros());
        }
        w.push(at(10), Vec3::x());
        assert_eq!(w.len(), 1);
        assert!(w.mean(at(10)).is_none());
    }
}
