//! [`FittedTimeBase`] — the L0 implementation of [`wslam_core::TimeBase`].
//!
//! Composes a camera [`CadenceModel`], a motion [`CadenceModel`] and an
//! [`OffsetFilter`]:
//!
//! ```text
//!   mapped_motion(k) = motion.predict(k)  - motion_origin
//!   mapped_camera(n) = camera.predict(n)  - camera_origin - td
//! ```
//!
//! Each stream is zeroed on its own first stamp, because nothing links the media
//! clock's epoch to the event loop's. That leaves exactly one unknown constant
//! between the two streams, and *that constant is `td`* — the residual offset
//! spec.md §4 L0 says to "estimate online as a filter state". It is not
//! manufactured here: [`FittedTimeBase::observe_offset`] is how a measurement
//! gets in, whether from the rig via [`crate::cross_correlate_lag`] or from an
//! online estimator above. Absent any measurement the offset stays at its prior
//! and [`TimeBase::is_converged`] stays false, which is the honest
//! answer — you cannot claim tight sync you never measured.

use wslam_core::{TimeBase, Timestamp};

use crate::cadence::{CadenceConfig, CadenceModel};
use crate::offset::OffsetFilter;

/// Configuration for [`FittedTimeBase`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClockConfig {
    /// Cadence model for `requestVideoFrameCallback` frames.
    pub camera: CadenceConfig,
    /// Cadence model for `DeviceMotion` events.
    pub motion: CadenceConfig,
    /// Prior mean for `td`, seconds. Zero unless a previous session or a rig run
    /// gave this device a calibrated value.
    pub initial_offset: f64,
    /// Prior variance for `td`, seconds squared.
    pub initial_offset_variance: f64,
    /// Random-walk variance added to `td` per epoch, seconds squared. See
    /// [`OffsetFilter::new`] for why it is per epoch rather than per second.
    pub offset_process_noise: f64,
    /// Hard clamp on the magnitude of the applied offset, seconds.
    ///
    /// Not a modelling choice — a backstop. A wrong `td` of a quarter second
    /// would push camera stamps past whole IMU windows, and the failure would
    /// surface as inexplicable tracking behaviour rather than as a clock bug.
    pub max_offset_seconds: f64,
    /// Reported offset variance at or below which the timebase declares itself
    /// converged, seconds squared.
    ///
    /// This is the gate on sensor tier 3 (spec.md §4, "tight visual-inertial ...
    /// Needs L0? yes"). The default corresponds to a 5 ms one-sigma residual
    /// offset — comfortably inside the 30 ms bar of spec.md §6 L0 while leaving
    /// no doubt that a stream we have not actually aligned stays out.
    pub converged_offset_variance: f64,
}

impl Default for ClockConfig {
    fn default() -> Self {
        ClockConfig {
            // Five seconds of history each: long enough that the prediction
            // noise at the newest index is a small fraction of the delivery
            // jitter, short enough to follow a real cadence change (thermal
            // throttling, a frame-rate renegotiation) within a few seconds.
            camera: CadenceConfig::for_rate(30.0, 5.0),
            motion: CadenceConfig::for_rate(60.0, 5.0),
            initial_offset: 0.0,
            // (50 ms)^2. Huai et al. measured up to 30 ms with native API access
            // (arXiv:2001.00470) and spec.md §5 warns the browser will be worse,
            // so the prior must not be tighter than that.
            initial_offset_variance: 2.5e-3,
            // (0.3 ms)^2 per epoch: two crystal oscillators drifting apart.
            offset_process_noise: 9.0e-8,
            max_offset_seconds: 0.25,
            converged_offset_variance: 2.5e-5,
        }
    }
}

/// The tier-3 timebase: cadence-fitted stamps plus an online camera-IMU offset.
#[derive(Debug, Clone)]
pub struct FittedTimeBase {
    config: ClockConfig,
    camera: CadenceModel,
    motion: CadenceModel,
    offset: OffsetFilter,
    camera_origin: Option<f64>,
    motion_origin: Option<f64>,
    last_camera_index: Option<u64>,
    last_motion_index: Option<u64>,
    last_camera_nanos: i64,
    last_motion_nanos: i64,
}

impl FittedTimeBase {
    /// Construct from configuration.
    #[must_use]
    pub fn new(config: ClockConfig) -> Self {
        let mut offset =
            OffsetFilter::new(config.initial_offset_variance, config.offset_process_noise);
        if config.initial_offset != 0.0 {
            offset.set_prior(config.initial_offset, config.initial_offset_variance);
        }
        FittedTimeBase {
            camera: CadenceModel::new(config.camera),
            motion: CadenceModel::new(config.motion),
            offset,
            config,
            camera_origin: None,
            motion_origin: None,
            last_camera_index: None,
            last_motion_index: None,
            last_camera_nanos: i64::MIN,
            last_motion_nanos: i64::MIN,
        }
    }

    /// Fold in a measurement of the camera-IMU offset, seconds.
    ///
    /// The only way `td` moves. On the rig this is
    /// [`crate::cross_correlate_lag`]'s `lag_seconds` and `variance`; online it
    /// is whatever residual-based estimator the orchestrator runs.
    pub fn observe_offset(&mut self, measured_offset: f64, variance: f64) {
        self.offset.update(measured_offset, variance);
    }

    /// Suspend or resume offset estimation (Li & Mourikis §V degenerate
    /// motions). Forwards to [`OffsetFilter::set_degenerate`], where the
    /// reasoning lives.
    pub fn set_degenerate(&mut self, degenerate: bool) {
        self.offset.set_degenerate(degenerate);
    }

    /// Age `td` by one epoch without a measurement, so a stretch with no offset
    /// observations widens the reported uncertainty instead of freezing it.
    pub fn propagate_offset(&mut self) {
        self.offset.propagate();
    }

    /// The camera cadence model, for reporting (spec.md §6 L0 wants the
    /// distribution of the jitter, and this is where it is measured).
    #[must_use]
    pub fn camera_cadence(&self) -> &CadenceModel {
        &self.camera
    }

    /// The motion cadence model.
    #[must_use]
    pub fn motion_cadence(&self) -> &CadenceModel {
        &self.motion
    }

    /// The offset filter.
    #[must_use]
    pub fn offset_filter(&self) -> &OffsetFilter {
        &self.offset
    }

    /// Configuration in force.
    #[must_use]
    pub fn config(&self) -> ClockConfig {
        self.config
    }

    /// Forget both streams and the offset. Called when capture restarts, so a
    /// previous session's epochs cannot leak into a new one.
    pub fn reset(&mut self) {
        self.camera.reset();
        self.motion.reset();
        self.offset.reset();
        if self.config.initial_offset != 0.0 {
            self.offset.set_prior(
                self.config.initial_offset,
                self.config.initial_offset_variance,
            );
        }
        self.camera_origin = None;
        self.motion_origin = None;
        self.last_camera_index = None;
        self.last_motion_index = None;
        self.last_camera_nanos = i64::MIN;
        self.last_motion_nanos = i64::MIN;
    }

    /// Emit a stamp, forbidding it to go backwards.
    ///
    /// Two things can step a mapped stream backwards: the switch from the raw
    /// fallback to the fitted line at convergence, and a fresh `td` observation
    /// shifting the camera stream under us. Both are one-off and sub-millisecond,
    /// and both would violate the ordering assumption that
    /// [`wslam_core::StateWindow`] and IMU interpolation are built on.
    ///
    /// This clamps the *output* only. The jitter measurement happens upstream in
    /// the [`CadenceModel`], which sees the raw stamps, so nothing is hidden
    /// from the number we report.
    fn emit(seconds: f64, last: &mut i64) -> Timestamp {
        let t = Timestamp::from_seconds(seconds);
        let n = t.nanos().max(*last);
        *last = n;
        Timestamp::from_nanos(n)
    }
}

impl Default for FittedTimeBase {
    fn default() -> Self {
        Self::new(ClockConfig::default())
    }
}

impl TimeBase for FittedTimeBase {
    // `_arrival_millis` is deliberately unused: L0 estimates the camera↔IMU
    // offset from cross-correlation (the `offset` model) rather than from
    // arrival stamps, whose event-loop jitter is the noise this layer exists
    // to remove. The parameter exists for the passthrough base, which has no
    // offset model and needs the arrival to share one origin across streams.
    fn map_camera(&mut self, media_time: f64, _arrival_millis: f64, frame_index: u64) -> Timestamp {
        self.camera.push(frame_index, media_time);
        let origin = *self.camera_origin.get_or_insert(media_time);
        // Before the model converges the raw stamp is all there is. mediaTime
        // rides the media clock, so it is a far better fallback than an arrival
        // stamp would be (spec.md §4 L0).
        let fitted = self.camera.predict(frame_index).unwrap_or(media_time);
        self.last_camera_index = Some(frame_index);
        // Positive td means camera stamps lag IMU stamps, so it comes off.
        let seconds = fitted - origin - self.camera_imu_offset();
        Self::emit(seconds, &mut self.last_camera_nanos)
    }

    fn map_motion(&mut self, event_index: u64, arrival_millis: f64) -> Timestamp {
        let arrival_seconds = arrival_millis * 1.0e-3;
        self.motion.push(event_index, arrival_seconds);
        let origin = *self.motion_origin.get_or_insert(arrival_seconds);
        let fitted = self.motion.predict(event_index).unwrap_or(arrival_seconds);
        self.last_motion_index = Some(event_index);
        // The IMU stream defines the timebase; the offset is applied to the
        // camera side only, so that a change in td cannot retime the inertial
        // history an orientation filter has already integrated.
        Self::emit(fitted - origin, &mut self.last_motion_nanos)
    }

    fn camera_imu_offset(&self) -> f64 {
        self.offset.offset().clamp(
            -self.config.max_offset_seconds,
            self.config.max_offset_seconds,
        )
    }

    fn offset_variance(&self) -> f64 {
        // Three independent contributions to the residual misalignment between a
        // camera stamp and an IMU stamp: how well we know td, and how well each
        // cadence model can place its own newest sample.
        let Some(camera_index) = self.last_camera_index else {
            return f64::INFINITY;
        };
        let Some(motion_index) = self.last_motion_index else {
            return f64::INFINITY;
        };
        self.offset.variance()
            + self.camera.prediction_variance(camera_index)
            + self.motion.prediction_variance(motion_index)
    }

    fn is_converged(&self) -> bool {
        self.camera.is_converged()
            && self.motion.is_converged()
            && self.offset_variance() <= self.config.converged_offset_variance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::correlate::cross_correlate_lag;
    use crate::synth::{CAMERA_30HZ, IMU_60HZ};
    use wslam_core::DeterministicRng;

    /// The rotation-rate programme the rig drives. Constant rate is useless for
    /// cross-correlation (every lag matches), so the turntable has to change
    /// rate — see `correlate`'s tests for the same point.
    fn rate(t: f64) -> f64 {
        1.2 * (2.0 * std::f64::consts::PI * 0.6 * t).sin()
            + 0.5 * (2.0 * std::f64::consts::PI * 2.1 * t + 0.7).sin()
            + 0.25 * (2.0 * std::f64::consts::PI * 4.7 * t + 2.0).sin()
    }

    /// Motion cadence used by the synthetic browser below.
    const MOTION_HZ: f64 = 60.0;
    /// Camera cadence used by the synthetic browser below.
    const CAMERA_HZ: f64 = 30.0;

    struct Recording {
        /// Mapped motion stamps, seconds, indexed by event index.
        motion: Vec<f64>,
        /// Mapped camera stamps *before* any offset correction, seconds.
        camera: Vec<f64>,
        /// True capture time of each camera frame, seconds.
        camera_truth: Vec<f64>,
        /// True sample time of each motion event, seconds.
        motion_truth: Vec<f64>,
    }

    /// Drive a `FittedTimeBase` with a synthetic browser.
    ///
    /// - motion events on a perfect 60 Hz hardware cadence, delivered with
    ///   one-sided heavy-tailed event-loop jitter;
    /// - camera frames on a perfect 30 Hz cadence, stamped with `mediaTime`
    ///   which is offset from the IMU clock by `td_true` and carries only the
    ///   small media-clock jitter;
    /// - the two epochs unrelated, as they are in a browser.
    fn record(tb: &mut FittedTimeBase, seed: u64, td_true: f64, seconds: f64) -> Recording {
        let mut rng = DeterministicRng::new("timebase-synth", seed);
        let motion_period = 1.0 / MOTION_HZ;
        let camera_period = 1.0 / CAMERA_HZ;
        // Unrelated epochs, and a camera stream that starts mid-way through a
        // motion interval so nothing lines up by accident.
        let motion_epoch = 1_234.5;
        let camera_epoch = 91_827.25;
        let camera_phase = 0.0071;

        let n_motion = (seconds / motion_period) as u64;
        let n_camera = (seconds / camera_period) as u64;

        let mut out = Recording {
            motion: Vec::new(),
            camera: Vec::new(),
            camera_truth: Vec::new(),
            motion_truth: Vec::new(),
        };

        // Interleave the two streams in true-time order, as a browser would.
        let (mut k, mut n) = (0u64, 0u64);
        while k < n_motion || n < n_camera {
            let t_m = k as f64 * motion_period;
            let t_c = camera_phase + n as f64 * camera_period;
            if k < n_motion && (n >= n_camera || t_m <= t_c) {
                let arrival_ms = (motion_epoch + t_m + IMU_60HZ.sample(&mut rng)) * 1.0e3;
                out.motion.push(tb.map_motion(k, arrival_ms).seconds());
                out.motion_truth.push(t_m);
                k += 1;
            } else {
                // mediaTime reads late by td_true for the same physical instant:
                // "camera stamps lag IMU stamps".
                let media = camera_epoch + t_c + td_true + CAMERA_30HZ.sample(&mut rng);
                out.camera
                    .push(tb.map_camera(media, media * 1e3, n).seconds());
                out.camera_truth.push(t_c);
                n += 1;
            }
        }
        out
    }

    /// Per-frame misalignment between the mapped camera stream and the mapped
    /// motion stream, in seconds, **before** any offset correction.
    ///
    /// For each camera frame this is the corrected-camera-minus-motion error the
    /// pipeline would actually see: the motion side is linearly interpolated
    /// between the two mapped events bracketing the frame's true capture
    /// instant, which is exactly how a consumer interpolates IMU samples
    /// ([`wslam_core::ImuSample::lerp`]), not a favourable abstraction of it.
    ///
    /// The **mean** of this is the constant the rig has to find. The **scatter**
    /// is the jitter that survived cadence fitting, and no offset estimate can
    /// remove it.
    fn misalignment(rec: &Recording, skip_frames: usize) -> Vec<f64> {
        let mut out = Vec::new();
        for (&mapped, &truth) in rec.camera.iter().zip(&rec.camera_truth).skip(skip_frames) {
            let k = truth * MOTION_HZ;
            let k0 = k.floor() as usize;
            if k0 + 1 >= rec.motion.len() {
                break;
            }
            let frac = k - k0 as f64;
            let reference = rec.motion[k0] * (1.0 - frac) + rec.motion[k0 + 1] * frac;
            out.push(mapped - reference);
        }
        out
    }

    /// The rig, in miniature: gyro angular rate stamped in the mapped motion
    /// timebase against image-derived rotation rate stamped in the mapped camera
    /// timebase, cross-correlated for the peak lag (spec.md §6 L0).
    ///
    /// Both signals carry sensor noise and **neither is built from ground
    /// truth timestamps**, so the recovered lag is an honest measurement of the
    /// constant separating the two streams.
    fn rig_measurement(rec: &Recording, seed: u64, skip: usize) -> crate::LagEstimate {
        let mut rng = DeterministicRng::new("rig-noise", seed);
        let skip_m = skip * 2;
        let gyro: Vec<(f64, f64)> = rec.motion[skip_m..]
            .iter()
            .zip(&rec.motion_truth[skip_m..])
            .map(|(&stamp, &truth)| (stamp, rate(truth) + rng.normal_with(0.0, 0.01)))
            .collect();
        let vision: Vec<(f64, f64)> = rec.camera[skip..]
            .iter()
            .zip(&rec.camera_truth[skip..])
            .map(|(&stamp, &truth)| (stamp, rate(truth) + rng.normal_with(0.0, 0.03)))
            .collect();
        cross_correlate_lag(&gyro, &vision, 0.20, 1.0 / MOTION_HZ).expect("rig peak")
    }

    #[test]
    fn beats_the_thirty_millisecond_bar_on_synthetic_jitter() {
        // spec.md §6 L0: "Bar: beat the 30 ms native-access figure from
        // arXiv:2001.00470." Huai et al. measured that with full native API
        // access; the browser has none of it.
        //
        // READ THE CRATE HEADER BEFORE QUOTING THIS NUMBER. A synthetic pass is
        // not the claim. The claim needs the turntable and the strobe, per
        // device. This test proves the estimator is correct on a jitter profile
        // we invented, and nothing more.
        const BAR_SECONDS: f64 = 0.030;

        // Note what happens to `td_true`: `map_camera` zeroes the stream on its
        // first `mediaTime`, so a constant media-clock offset is absorbed by the
        // origin and is not separately observable. What the rig measures — and
        // what the offset filter carries — is the *total* constant between the
        // two zeroed streams: the media-clock offset, plus the difference in
        // typical delivery delay between the two paths, plus the two arbitrary
        // first-sample delays. That total is the only thing anything downstream
        // cares about, which is why the estimate is checked against it below
        // rather than against `td_true`.
        //
        // Several seeds, and every one has to clear the bar. A single draw of a
        // heavy-tailed process is an anecdote, and "report the distribution, not
        // a mean" (spec.md §6 L0) applies to our own test evidence too.
        let trials: [(u64, f64); 5] = [
            (0xB0_0B, 0.0247),
            (0x1234, -0.0180),
            (0xFACE, 0.0),
            (0x9911, 0.0620),
            (0x0042, -0.0455),
        ];

        for (seed, td_true) in trials {
            let mut tb = FittedTimeBase::new(ClockConfig::default());
            let rec = record(&mut tb, seed, td_true, 30.0);

            assert!(tb.camera_cadence().is_converged());
            assert!(tb.motion_cadence().is_converged());

            let warm = rec.camera.len() / 8;
            let lag = rig_measurement(&rec, seed ^ 0x5EED, warm);
            tb.observe_offset(lag.lag_seconds, lag.variance.max(1.0e-8));
            let td_est = tb.camera_imu_offset();

            // --- residual offset, the metric spec.md §6 L0 names -------------
            let raw = misalignment(&rec, warm);
            assert!(raw.len() > 500);
            let truth_constant = raw.iter().sum::<f64>() / raw.len() as f64;
            let uncorrected = (raw.iter().map(|r| r * r).sum::<f64>() / raw.len() as f64).sqrt();
            let residuals: Vec<f64> = raw.iter().map(|m| m - td_est).collect();

            let bias = residuals.iter().sum::<f64>() / residuals.len() as f64;
            let rms =
                (residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64).sqrt();
            let abs: Vec<f64> = residuals.iter().map(|r| r.abs()).collect();
            let p95 = wslam_core::stats::percentile(&abs, 0.95).unwrap();
            let worst = abs.iter().cloned().fold(0.0, f64::max);
            eprintln!(
                "L0 seed {seed:#x}: residual offset bias {:.3} ms, rms {:.3} ms, \
                 p95 {:.3} ms, worst {:.3} ms; constant {:.3} ms recovered as {:.3} ms",
                bias * 1e3,
                rms * 1e3,
                p95 * 1e3,
                worst * 1e3,
                truth_constant * 1e3,
                td_est * 1e3
            );

            // The distribution, not just the mean — spec.md §6 L0 is explicit.
            assert!(
                rms < 0.1 * BAR_SECONDS,
                "seed {seed:#x}: residual offset rms {:.3} ms against the {:.0} ms bar",
                rms * 1e3,
                BAR_SECONDS * 1e3
            );
            assert!(
                p95 < 0.15 * BAR_SECONDS,
                "seed {seed:#x}: p95 {:.3} ms",
                p95 * 1e3
            );
            assert!(
                worst < 0.5 * BAR_SECONDS,
                "seed {seed:#x}: worst-case {:.3} ms",
                worst * 1e3
            );
            assert!(
                bias.abs() < 0.05 * BAR_SECONDS,
                "seed {seed:#x}: bias {:.3} ms",
                bias * 1e3
            );

            // The rig measurement must have found the constant, or the residual
            // is small for the wrong reason.
            assert!(
                (td_est - truth_constant).abs() < 0.05 * BAR_SECONDS,
                "seed {seed:#x}: recovered {:.3} ms vs constant {:.3} ms",
                td_est * 1e3,
                truth_constant * 1e3
            );

            // The control arm, in-line: uncorrected, this trial would have been
            // nowhere near the bar. Without it, a run that happened to start
            // already aligned would pass everything above for no reason.
            assert!(
                uncorrected > 5.0 * rms,
                "seed {seed:#x}: uncorrected misalignment {:.3} ms is too close to the \
                 corrected {:.3} ms for the correction to be doing the work",
                uncorrected * 1e3,
                rms * 1e3
            );

            // And the number we would publish alongside it.
            assert!(tb.offset_variance().is_finite());
            assert!(
                tb.offset_variance().sqrt() < 0.05 * BAR_SECONDS,
                "seed {seed:#x}: reported sigma {:.3} ms",
                tb.offset_variance().sqrt() * 1e3
            );
        }
    }

    #[test]
    fn reported_offset_variance_is_not_a_fantasy() {
        // A calibrated uncertainty is the whole product (spec.md §1). If the
        // reported sigma does not bracket the actual residual, it is decoration.
        //
        // The bracket is 3 sigma of the *reported* variance plus the correlated
        // -residual optimism documented on `LagEstimate::variance`: the rig
        // signals are resampled at 60 Hz while the vision stream is 30 Hz, so
        // the correlation's own variance is optimistic by roughly the
        // oversampling factor. Being explicit about that here is better than
        // quietly widening the bound.
        let td_true = -0.0180; // camera ahead of the IMU, the other sign
        let mut tb = FittedTimeBase::new(ClockConfig::default());
        let rec = record(&mut tb, 4242, td_true, 30.0);

        let warm = rec.camera.len() / 8;
        let lag = rig_measurement(&rec, 5, warm);
        tb.observe_offset(lag.lag_seconds, lag.variance.max(1.0e-8));

        let raw = misalignment(&rec, warm);
        let truth_constant = raw.iter().sum::<f64>() / raw.len() as f64;
        let error = (tb.camera_imu_offset() - truth_constant).abs();
        let sigma = tb.offset_variance().sqrt();
        eprintln!(
            "L0 offset error {:.4} ms against reported sigma {:.4} ms",
            error * 1e3,
            sigma * 1e3
        );
        assert!(
            error < 3.0 * std::f64::consts::SQRT_2 * sigma,
            "residual {:.4} ms exceeds the bracket around sigma {:.4} ms — \
             the covariance is overconfident",
            error * 1e3,
            sigma * 1e3
        );
    }

    #[test]
    fn offset_is_subtracted_from_the_camera_stream_only() {
        // The sign convention, pinned. wslam_core::TimeBase: "Positive means
        // camera stamps lag IMU stamps", so a positive td must pull camera
        // stamps earlier and leave motion stamps untouched.
        let mut tb = FittedTimeBase::new(ClockConfig::default());
        for k in 0..600u64 {
            tb.map_motion(k, 1000.0 + k as f64 * (1000.0 / 60.0));
        }
        for n in 0..300u64 {
            tb.map_camera(50.0 + n as f64 / 30.0, (50.0 + n as f64 / 30.0) * 1e3, n);
        }
        let camera_before = tb
            .map_camera(50.0 + 300.0 / 30.0, (50.0 + 300.0 / 30.0) * 1e3, 300)
            .seconds();
        let motion_before = tb
            .map_motion(600, 1000.0 + 600.0 * (1000.0 / 60.0))
            .seconds();

        tb.observe_offset(0.020, 1.0e-8);
        assert!(tb.camera_imu_offset() > 0.0);
        let camera_after = tb
            .map_camera(50.0 + 301.0 / 30.0, (50.0 + 301.0 / 30.0) * 1e3, 301)
            .seconds();
        let motion_after = tb
            .map_motion(601, 1000.0 + 601.0 * (1000.0 / 60.0))
            .seconds();

        // One extra frame period elapsed, minus the newly applied 20 ms offset.
        let camera_step = camera_after - camera_before;
        assert!(
            (camera_step - (1.0 / 30.0 - 0.020)).abs() < 1e-6,
            "camera advanced by {camera_step}"
        );
        let motion_step = motion_after - motion_before;
        assert!(
            (motion_step - 1.0 / 60.0).abs() < 1e-9,
            "motion stream must not move: {motion_step}"
        );
    }

    #[test]
    fn offset_is_clamped_to_a_sane_range() {
        let mut tb = FittedTimeBase::new(ClockConfig::default());
        tb.observe_offset(3.0, 1.0e-12);
        assert!((tb.camera_imu_offset() - 0.25).abs() < 1e-15);
        tb.observe_offset(-3.0, 1.0e-12);
        assert!((tb.camera_imu_offset() + 0.25).abs() < 1e-15);
    }

    #[test]
    fn mapped_stamps_are_non_decreasing_through_convergence() {
        // The raw-to-fitted switch and a mid-stream td observation are the two
        // places a mapped stream could step backwards. Neither may.
        let mut tb = FittedTimeBase::new(ClockConfig::default());
        let mut rng = DeterministicRng::new("monotone", 3);
        let mut last_m = Timestamp::from_nanos(i64::MIN);
        let mut last_c = Timestamp::from_nanos(i64::MIN);
        for i in 0..1200u64 {
            let t = tb.map_motion(
                i,
                (1000.0 + i as f64 / 60.0 + IMU_60HZ.sample(&mut rng)) * 1e3,
            );
            assert!(t >= last_m, "motion went backwards at {i}");
            last_m = t;
            if i % 2 == 0 {
                let n = i / 2;
                let media = 7.0 + n as f64 / 30.0 + CAMERA_30HZ.sample(&mut rng);
                let t = tb.map_camera(media, media * 1e3, n);
                assert!(t >= last_c, "camera went backwards at frame {n}");
                last_c = t;
            }
            if i == 800 {
                tb.observe_offset(0.030, 1.0e-8);
            }
        }
    }

    #[test]
    fn first_sample_of_each_stream_is_the_origin() {
        let mut tb = FittedTimeBase::new(ClockConfig::default());
        assert_eq!(tb.map_camera(12_345.678, 12_345_678.0, 0), Timestamp::ZERO);
        assert_eq!(tb.map_motion(0, 98_765.4), Timestamp::ZERO);
    }

    #[test]
    fn falls_back_to_raw_stamps_before_convergence() {
        // Tier 3 is gated, but the timebase still has to return *something*
        // usable from frame zero, and the raw stamp is the best available.
        let mut tb = FittedTimeBase::new(ClockConfig::default());
        tb.map_camera(100.0, 100_000.0, 0);
        let t = tb.map_camera(100.25, 100_250.0, 1);
        assert!((t.seconds() - 0.25).abs() < 1e-9);
        assert!(!tb.is_converged());
    }

    #[test]
    fn offset_variance_is_infinite_until_both_streams_have_spoken() {
        let mut tb = FittedTimeBase::new(ClockConfig::default());
        assert!(tb.offset_variance().is_infinite());
        assert!(!tb.is_converged());
        for k in 0..600u64 {
            tb.map_motion(k, 1000.0 + k as f64 * (1000.0 / 60.0));
        }
        // Motion alone is not enough: there is no camera index to predict at.
        assert!(tb.offset_variance().is_infinite());
        assert!(!tb.is_converged());
    }

    #[test]
    fn convergence_gates_on_the_offset_being_measured_not_merely_assumed() {
        // Both cadence models converge on clean data almost immediately. Tier 3
        // must still stay closed until somebody has actually measured td,
        // because an unmeasured offset is not a small offset.
        let mut tb = FittedTimeBase::new(ClockConfig::default());
        for k in 0..600u64 {
            tb.map_motion(k, 1000.0 + k as f64 * (1000.0 / 60.0));
        }
        for n in 0..300u64 {
            tb.map_camera(50.0 + n as f64 / 30.0, (50.0 + n as f64 / 30.0) * 1e3, n);
        }
        assert!(tb.camera_cadence().is_converged());
        assert!(tb.motion_cadence().is_converged());
        assert!(
            !tb.is_converged(),
            "converged with the prior offset variance still in place"
        );

        tb.observe_offset(0.021, 1.0e-8);
        assert!(tb.is_converged());
    }

    #[test]
    fn a_degenerate_stretch_reopens_the_gate() {
        // Li & Mourikis §V: under degenerate motion the offset stops being
        // observable. The variance must grow back through the tier-3 threshold
        // rather than sit at its last confident value.
        let mut tb = FittedTimeBase::new(ClockConfig::default());
        for k in 0..600u64 {
            tb.map_motion(k, 1000.0 + k as f64 * (1000.0 / 60.0));
        }
        for n in 0..300u64 {
            tb.map_camera(50.0 + n as f64 / 30.0, (50.0 + n as f64 / 30.0) * 1e3, n);
        }
        tb.observe_offset(0.021, 1.0e-9);
        assert!(tb.is_converged());
        let held = tb.camera_imu_offset();

        tb.set_degenerate(true);
        for _ in 0..100_000 {
            tb.propagate_offset();
        }
        assert!(
            !tb.is_converged(),
            "a suspended filter must not claim tier 3"
        );
        assert_eq!(
            tb.camera_imu_offset().to_bits(),
            held.to_bits(),
            "the estimate must not drift while suspended"
        );

        tb.set_degenerate(false);
        for _ in 0..50 {
            tb.observe_offset(0.021, 1.0e-9);
        }
        assert!(tb.is_converged());
    }

    #[test]
    fn a_configured_prior_is_applied_from_the_first_frame() {
        let config = ClockConfig {
            initial_offset: 0.018,
            initial_offset_variance: 1.0e-6,
            ..ClockConfig::default()
        };
        let tb = FittedTimeBase::new(config);
        assert!((tb.camera_imu_offset() - 0.018).abs() < 1e-15);
        assert!((tb.offset_filter().variance() - 1.0e-6).abs() < 1e-18);
    }

    #[test]
    fn reset_restores_the_prior_and_forgets_both_epochs() {
        let mut tb = FittedTimeBase::new(ClockConfig::default());
        record(&mut tb, 1, 0.02, 5.0);
        tb.observe_offset(0.02, 1.0e-8);
        tb.reset();
        assert_eq!(tb.camera_imu_offset(), 0.0);
        assert!(tb.offset_variance().is_infinite());
        assert!(!tb.is_converged());
        assert_eq!(tb.map_camera(500.0, 500_000.0, 0), Timestamp::ZERO);
    }

    #[test]
    fn replay_is_bit_exact() {
        // spec.md §6: "the same binary then runs live *and* replays a canned
        // trajectory bit-for-bit reproducibly."
        let run = || {
            let mut tb = FittedTimeBase::new(ClockConfig::default());
            let rec = record(&mut tb, 777, 0.0215, 12.0);
            (rec.camera, rec.motion, tb.offset_variance())
        };
        let (c1, m1, v1) = run();
        let (c2, m2, v2) = run();
        assert_eq!(c1.len(), c2.len());
        for (a, b) in c1.iter().zip(&c2) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        for (a, b) in m1.iter().zip(&m2) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        assert_eq!(v1.to_bits(), v2.to_bits());
    }

    #[test]
    fn is_send() {
        // wslam_core::TimeBase requires Send; the orchestrator moves the clock
        // onto the frontend thread.
        fn assert_send<T: Send>() {}
        assert_send::<FittedTimeBase>();
    }
}
