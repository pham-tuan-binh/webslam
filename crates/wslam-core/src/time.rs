//! The unified timebase (L0's output contract).
//!
//! spec.md §6 is unambiguous: *"No `Date.now()` or `performance.now()` anywhere
//! in the pipeline. Every timestamp enters through the clock layer."* This
//! module is how that rule is enforced structurally — there is no function here
//! that reads a wall clock, and no pipeline type takes a `HostClock`.

use std::fmt;

/// Nanoseconds in the unified timebase. Integral, so replay is bit-exact.
pub type Nanos = i64;

/// A point in the unified timebase.
///
/// The origin is arbitrary but fixed per session: for live capture it is the
/// first `performance.now()` sampled *by the shim*, mapped in once, never read
/// again. For replay it is the dataset's own origin. Nothing downstream may
/// assume it corresponds to any external epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Timestamp(Nanos);

impl Timestamp {
    /// The timebase origin.
    pub const ZERO: Timestamp = Timestamp(0);

    /// Construct from nanoseconds.
    #[inline]
    #[must_use]
    pub const fn from_nanos(n: Nanos) -> Self {
        Timestamp(n)
    }

    /// Construct from seconds. Rounds to the nearest nanosecond.
    #[inline]
    #[must_use]
    pub fn from_seconds(s: f64) -> Self {
        Timestamp((s * 1.0e9).round() as Nanos)
    }

    /// Construct from milliseconds — the unit the browser hands us.
    #[inline]
    #[must_use]
    pub fn from_millis_f64(ms: f64) -> Self {
        Timestamp((ms * 1.0e6).round() as Nanos)
    }

    /// Nanoseconds since the timebase origin.
    #[inline]
    #[must_use]
    pub const fn nanos(self) -> Nanos {
        self.0
    }

    /// Seconds since the timebase origin.
    #[inline]
    #[must_use]
    pub fn seconds(self) -> f64 {
        self.0 as f64 * 1.0e-9
    }

    /// Milliseconds since the timebase origin — what crosses into JS.
    #[inline]
    #[must_use]
    pub fn millis(self) -> f64 {
        self.0 as f64 * 1.0e-6
    }

    /// Signed interval `self - earlier`, in seconds.
    #[inline]
    #[must_use]
    pub fn since(self, earlier: Timestamp) -> f64 {
        (self.0 - earlier.0) as f64 * 1.0e-9
    }

    /// Shift by a signed number of seconds.
    #[inline]
    #[must_use]
    pub fn offset_seconds(self, dt: f64) -> Self {
        Timestamp(self.0 + (dt * 1.0e9).round() as Nanos)
    }

    /// Shift by a signed number of nanoseconds.
    #[inline]
    #[must_use]
    pub fn offset_nanos(self, dt: Nanos) -> Self {
        Timestamp(self.0 + dt)
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.6}s", self.seconds())
    }
}

/// Maps sensor-native stamps into the unified timebase.
///
/// This is the contract L0 (`wslam-clock`) fulfils, and the seam that lets tier
/// 1 and 2 run without L0 at all (spec.md §4: "L0 is off the critical path").
///
/// Implementations must be **pure functions of their arguments plus internal
/// filter state**. They may not consult a wall clock.
pub trait TimeBase: Send {
    /// Map a camera frame.
    ///
    /// `media_time` is `VideoFrameCallbackMetadata.mediaTime` in seconds — it
    /// rides the media clock, not the wall clock, which is exactly why the shim
    /// forwards it raw (spec.md §4 L0). `arrival_millis` is the shim's raw
    /// arrival stamp for the same frame, in the same clock the motion stream's
    /// arrivals are stamped in — it is what lets an implementation put the two
    /// streams on **one** origin instead of zeroing each independently.
    /// `frame_index` is the monotonically increasing index of frames
    /// *delivered to us*, used to fit the cadence model that per-event
    /// timestamps are too jittery to support.
    fn map_camera(&mut self, media_time: f64, arrival_millis: f64, frame_index: u64) -> Timestamp;

    /// Map a motion event.
    ///
    /// `event_index` is the delivery index; `arrival_millis` is the shim's raw
    /// arrival stamp. The linear model is fit over *index*, because
    /// `DeviceMotion` events are generated on a regular hardware cadence and
    /// only delivered with event-loop jitter.
    fn map_motion(&mut self, event_index: u64, arrival_millis: f64) -> Timestamp;

    /// Current best estimate of the camera-to-IMU temporal offset, in seconds.
    /// Positive means camera stamps lag IMU stamps. Zero for a passthrough base.
    fn camera_imu_offset(&self) -> f64 {
        0.0
    }

    /// Variance of the residual offset, in seconds squared. This — not the mean
    /// — is the number spec.md §6 L0 asks us to report.
    fn offset_variance(&self) -> f64 {
        f64::INFINITY
    }

    /// Whether the model has seen enough samples to be trusted.
    fn is_converged(&self) -> bool {
        false
    }
}

/// The tier-1/tier-2 timebase: trust the stamps as delivered.
///
/// Correct for loose coupling, which needs ordering and approximate alignment
/// but not sub-frame accuracy. Tier 3 swaps in `wslam_clock::FittedTimeBase`.
///
/// Both streams share **one** origin: the arrival stamp of whichever sample
/// shows up first. An earlier version zeroed each stream at its own first
/// sample, which silently baked the stream *start-order* into every
/// cross-stream lookup — the motion listener starts before the camera
/// delivers, so every `attitude_at(frame_time)` interpolated attitude from
/// 100–300 ms in the past. Camera timestamps advance on the media clock
/// (low jitter) but are anchored at the first frame's arrival; motion uses
/// arrivals directly. The residual cross-stream error is the camera pipeline's
/// capture-to-delivery latency on that first frame — bounded, and the thing
/// L0 exists to remove at tier 3.
#[derive(Debug, Clone, Default)]
pub struct PassthroughTimeBase {
    /// Arrival stamp of the first sample seen on *either* stream, ms.
    origin_arrival: Option<f64>,
    /// First camera frame: `(media_time, anchor_seconds)` where the anchor is
    /// that frame's arrival relative to `origin_arrival`.
    camera_anchor: Option<(f64, f64)>,
}

impl PassthroughTimeBase {
    /// Construct an unstarted passthrough timebase.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl TimeBase for PassthroughTimeBase {
    fn map_camera(&mut self, media_time: f64, arrival_millis: f64, _frame_index: u64) -> Timestamp {
        let origin = *self.origin_arrival.get_or_insert(arrival_millis);
        let (media_first, anchor_s) = *self
            .camera_anchor
            .get_or_insert((media_time, (arrival_millis - origin) / 1000.0));
        Timestamp::from_seconds(anchor_s + (media_time - media_first))
    }

    fn map_motion(&mut self, _event_index: u64, arrival_millis: f64) -> Timestamp {
        let origin = *self.origin_arrival.get_or_insert(arrival_millis);
        Timestamp::from_millis_f64(arrival_millis - origin)
    }

    fn is_converged(&self) -> bool {
        true // it has nothing to converge to
    }
}

/// Off-pipeline wall clock, for profiling only.
///
/// Deliberately a separate trait from [`TimeBase`] so that "did this code read
/// the wall clock?" is answerable by grepping for one name. Only the debug
/// timings surface (`slam.debug.timings()`) may hold one. **Anything on the
/// estimation path that takes a `HostClock` is a bug**, and a review rule.
pub trait HostClock: Send {
    /// Elapsed wall time in seconds since an arbitrary fixed origin.
    fn elapsed_seconds(&self) -> f64;
}

/// Native `std::time`-backed host clock. Not available on wasm.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone)]
pub struct StdHostClock {
    origin: std::time::Instant,
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for StdHostClock {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl StdHostClock {
    /// Start a host clock at the current instant.
    #[must_use]
    pub fn new() -> Self {
        StdHostClock {
            origin: std::time::Instant::now(),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl HostClock for StdHostClock {
    fn elapsed_seconds(&self) -> f64 {
        self.origin.elapsed().as_secs_f64()
    }
}

/// A host clock that advances only when told to. Used by tests and replay so
/// that timing-dependent debug output is reproducible.
#[derive(Debug, Clone, Default)]
pub struct ManualHostClock {
    seconds: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl ManualHostClock {
    /// Start at zero.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance by `dt` seconds.
    pub fn advance(&self, dt: f64) {
        use std::sync::atomic::Ordering;
        let cur = f64::from_bits(self.seconds.load(Ordering::Relaxed));
        self.seconds.store((cur + dt).to_bits(), Ordering::Relaxed);
    }
}

impl HostClock for ManualHostClock {
    fn elapsed_seconds(&self) -> f64 {
        f64::from_bits(self.seconds.load(std::sync::atomic::Ordering::Relaxed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nanosecond_roundtrip_is_exact() {
        let t = Timestamp::from_nanos(1_234_567_891);
        assert_eq!(t.nanos(), 1_234_567_891);
        assert_eq!(Timestamp::from_seconds(t.seconds()).nanos(), t.nanos());
    }

    #[test]
    fn millis_conversion_matches_browser_units() {
        let t = Timestamp::from_millis_f64(16.6667);
        assert_eq!(t.nanos(), 16_666_700);
        assert!((t.millis() - 16.6667).abs() < 1e-9);
    }

    #[test]
    fn passthrough_zeroes_at_the_first_sample_of_either_stream() {
        let mut tb = PassthroughTimeBase::new();
        assert_eq!(tb.map_camera(1000.5, 7_000.0, 0), Timestamp::ZERO);
        assert_eq!(
            tb.map_camera(1000.75, 7_020.0, 1),
            Timestamp::from_seconds(0.25)
        );
        // Motion arriving later lands *later* on the shared clock — not at a
        // fresh zero of its own.
        assert_eq!(tb.map_motion(0, 7_005.0), Timestamp::from_millis_f64(5.0));
        assert_eq!(tb.map_motion(1, 7_015.0), Timestamp::from_millis_f64(15.0));
    }

    #[test]
    fn passthrough_keeps_the_streams_on_one_origin_regardless_of_start_order() {
        // The browser's actual order: motion permission resolves and samples
        // flow ~200 ms before the first frame is delivered. The frame must
        // land at ~200 ms on the shared clock — the old per-stream zeroing put
        // it at 0, so every attitude lookup was 200 ms stale.
        let mut tb = PassthroughTimeBase::new();
        assert_eq!(tb.map_motion(0, 3_000.0), Timestamp::ZERO);
        assert_eq!(tb.map_motion(1, 3_100.0), Timestamp::from_millis_f64(100.0));
        let first = tb.map_camera(42.0, 3_200.0, 0);
        assert!((first.seconds() - 0.2).abs() < 1e-9, "{first}");
        // Subsequent frames advance on the media clock from that anchor.
        let second = tb.map_camera(42.0333, 3_233.0, 1);
        assert!((second.seconds() - 0.2333).abs() < 1e-9, "{second}");
    }

    #[test]
    fn passthrough_camera_spacing_rides_the_media_clock_not_arrivals() {
        // Delivery jitter on later frames must not reach the timestamps; only
        // the first frame anchors, everything after advances by mediaTime.
        let mut tb = PassthroughTimeBase::new();
        tb.map_camera(10.0, 1_000.0, 0);
        // 33.3 ms of media time, but the callback fired 80 ms late.
        let t = tb.map_camera(10.0333, 1_113.0, 1);
        assert!((t.seconds() - 0.0333).abs() < 1e-9, "{t}");
    }

    #[test]
    fn since_is_signed() {
        let a = Timestamp::from_seconds(1.0);
        let b = Timestamp::from_seconds(2.5);
        assert!((b.since(a) - 1.5).abs() < 1e-12);
        assert!((a.since(b) + 1.5).abs() < 1e-12);
    }

    #[test]
    fn manual_host_clock_only_moves_when_told() {
        let c = ManualHostClock::new();
        assert_eq!(c.elapsed_seconds(), 0.0);
        c.advance(0.25);
        c.advance(0.25);
        assert!((c.elapsed_seconds() - 0.5).abs() < 1e-15);
    }
}
