//! The error-state Kalman filter itself.

use nalgebra::Matrix3x6;
use wslam_core::covariance::symmetrize;
use wslam_core::imu::{ImuSample, GRAVITY};
use wslam_core::math::{hat, Mat3, Mat6, Scalar, So3, Vec3, Vec6};
use wslam_core::time::Timestamp;

use crate::config::OrientationConfig;
use crate::gravity::{
    gravity_direction_body, rotation_aligning, world_up, wrap_angle, yaw_axis_body,
    yaw_jacobian_body, yaw_of,
};
use crate::history::{AttitudeHistory, HISTORY_CAPACITY};
use crate::rates::RateWindow;

/// Renormalise the attitude quaternion this often.
///
/// Unit-quaternion products drift off the unit sphere at ~1e-16 per multiply, so
/// a 60 s session at 100 Hz accumulates ~1e-13 — harmless, but the cost of
/// bounding it is one division per 64 samples and the alternative is a slow
/// scale error that only manifests in long runs, which is precisely the class of
/// bug a 15-minute thermal soak (spec.md §6, System level) would surface late.
const RENORMALIZE_INTERVAL: u32 = 64;

/// Sample gaps longer than this mean the event stream stalled.
const MAX_TRUSTED_DT: Scalar = 0.2;

/// Prior variance on turn-on gyro bias, (rad/s)^2. One sigma is ~1 deg/s, which
/// is the order of magnitude a consumer MEMS gyro powers up with.
const BIAS_PRIOR_VARIANCE: Scalar = 3.0e-4;

/// Initial variance along the yaw direction, rad^2.
///
/// Gravity says nothing about heading, so at initialisation yaw is uniform on
/// the circle and its variance is `pi^2/3`. Reporting a small number here would
/// be the exact failure spec.md §6 L6 calls *"overconfident, which is worse than
/// no covariance at all"* — and L3 needs to see a large yaw variance to know its
/// own yaw observation should dominate.
const INITIAL_YAW_VARIANCE: Scalar = std::f64::consts::PI * std::f64::consts::PI / 3.0;

/// Counters for the debug surface and for tests that need to prove a gate fired.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FilterStats {
    /// Samples integrated.
    pub samples: u64,
    /// Samples discarded as non-finite, duplicated or out of order.
    pub rejected_samples: u64,
    /// Accelerometer updates applied.
    pub gravity_accepted: u64,
    /// Accelerometer updates refused by the magnitude gate.
    pub gravity_rejected: u64,
    /// Zero-angular-rate bias updates applied.
    pub static_updates: u64,
    /// Yaw corrections accepted from L3.
    pub yaw_corrections: u64,
    /// Updates abandoned because the innovation covariance was singular.
    pub numerical_failures: u64,
}

/// L1: gyro integration with accelerometer gravity correction.
///
/// An error-state Kalman filter on SO(3) with the gyro bias in the state, not a
/// complementary filter. The distinction is not stylistic: the covariance this
/// produces is what L6 publishes and what spec.md §6 L6 promises to validate by
/// NEES, and a hand-tuned blend coefficient has no covariance to publish.
///
/// State, in `[attitude error; gyro bias error]` order:
///
/// - nominal attitude `R_world_body` and gyro bias `b`, carried outside the
///   linear filter;
/// - error state `[δθ; δb]` in R^6 with `R_true = R_est · exp(δθ)` and
///   `b_true = b + δb`, whose mean is injected and reset to zero every update.
///
/// Observability, which drives every design choice below: gravity fixes roll and
/// pitch and *nothing else, ever*. Yaw is unobservable to L1 for all time — it
/// drifts on the gyro bias and is arrested by L3 through
/// [`OrientationFilter::correct_yaw`].
#[derive(Debug, Clone)]
pub struct OrientationFilter {
    config: OrientationConfig,
    attitude: So3,
    bias: Vec3,
    covariance: Mat6,
    previous: Option<ImuSample>,
    history: AttitudeHistory,
    rates: RateWindow,
    initialized: bool,
    since_renormalize: u32,
    stats: FilterStats,
}

impl OrientationFilter {
    /// Construct an uninitialised filter. The first accelerometer sample inside
    /// the gravity gate sets the attitude and starts estimation.
    #[must_use]
    pub fn new(config: OrientationConfig) -> Self {
        OrientationFilter {
            config: config.sanitized(),
            attitude: So3::identity(),
            bias: Vec3::zeros(),
            covariance: Mat6::identity(),
            previous: None,
            history: AttitudeHistory::new(HISTORY_CAPACITY),
            rates: RateWindow::new(crate::rates::STATIC_WINDOW_SECONDS),
            initialized: false,
            since_renormalize: 0,
            stats: FilterStats::default(),
        }
    }

    /// Fold one inertial sample into the estimate.
    ///
    /// Propagate on the gyro, then correct on gravity, then correct the bias if
    /// the device is holding still. Samples that are non-finite, duplicated or
    /// out of order are counted and dropped: the shim does no reordering
    /// (spec.md §7) but the queues above it can still deliver a stale event, and
    /// integrating one twice injects rotation that never happened.
    pub fn integrate(&mut self, sample: &ImuSample) {
        if !is_finite(&sample.gyro) || !is_finite(&sample.accel) {
            self.stats.rejected_samples += 1;
            return;
        }

        let step = match self.previous {
            Some(previous) => {
                let dt = sample.timestamp.since(previous.timestamp);
                if dt <= 0.0 {
                    self.stats.rejected_samples += 1;
                    return;
                }
                Some((previous, dt))
            }
            None => None,
        };

        let just_initialized = if self.initialized {
            if let Some((previous, dt)) = step {
                self.propagate(&previous, sample, dt);
            }
            false
        } else {
            self.try_initialize(sample)
        };

        self.previous = Some(*sample);
        self.stats.samples += 1;

        if self.initialized {
            // The sample that set the attitude has already been consumed;
            // correcting with it again would double-count one measurement.
            if !just_initialized {
                self.update_gravity(sample);
            }
            if let Some((_, dt)) = step {
                self.update_zero_rate(sample, dt);
            }
            self.renormalize_if_due();
            self.history.push(sample.timestamp, self.attitude);
        }

        // After the updates, so that the stillness detector's window holds only
        // samples older than the one it was asked to vouch for. Before
        // initialisation too, so the window is already warm when the first
        // gravity sample arrives.
        self.rates.push(sample.timestamp, sample.gyro);
    }

    /// Current attitude, `R_world_body`. Identity until initialised.
    #[inline]
    #[must_use]
    pub fn attitude(&self) -> So3 {
        self.attitude
    }

    /// Unit vector along gravitational acceleration, in body coordinates.
    ///
    /// Points **down**: a level device gives `(0, 0, -1)`. This is the negation
    /// of the direction the accelerometer reads at rest.
    #[inline]
    #[must_use]
    pub fn gravity_body(&self) -> Vec3 {
        gravity_direction_body(&self.attitude)
    }

    /// Estimated gyroscope bias, rad/s, body frame.
    #[inline]
    #[must_use]
    pub fn gyro_bias(&self) -> Vec3 {
        self.bias
    }

    /// Attitude error covariance, rad^2, in the body frame under the
    /// right-perturbation convention.
    ///
    /// Expect the yaw direction — `gravity_body()`, up to sign — to carry a very
    /// large variance and to grow monotonically until L3 supplies a heading.
    /// That is the honest statement of what L1 knows.
    #[inline]
    #[must_use]
    pub fn covariance(&self) -> Mat3 {
        self.covariance.fixed_view::<3, 3>(0, 0).into_owned()
    }

    /// Gyro bias covariance, (rad/s)^2.
    #[inline]
    #[must_use]
    pub fn bias_covariance(&self) -> Mat3 {
        self.covariance.fixed_view::<3, 3>(3, 3).into_owned()
    }

    /// Full `[attitude; bias]` covariance, including the cross terms.
    ///
    /// The cross terms are the whole reason the bias is estimable at all, and
    /// they are what L6 needs to propagate L1's uncertainty forward honestly.
    #[inline]
    #[must_use]
    pub fn full_covariance(&self) -> Mat6 {
        self.covariance
    }

    /// Whether an attitude has been established from gravity.
    #[inline]
    #[must_use]
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// World yaw, radians, from the tilt-twist decomposition. See
    /// [`crate::gravity::yaw_of`].
    #[inline]
    #[must_use]
    pub fn yaw(&self) -> Scalar {
        yaw_of(&self.attitude)
    }

    /// Timestamp of the last accepted sample.
    #[inline]
    #[must_use]
    pub fn last_timestamp(&self) -> Option<Timestamp> {
        self.previous.map(|s| s.timestamp)
    }

    /// Counters for the debug surface.
    #[inline]
    #[must_use]
    pub fn stats(&self) -> FilterStats {
        self.stats
    }

    /// Return to the uninitialised state, keeping the configuration.
    pub fn reset(&mut self) {
        let config = self.config;
        *self = OrientationFilter::new(config);
    }

    /// Absolute yaw observation from L3, arresting yaw drift.
    ///
    /// `yaw_world` is a heading in radians about world `+Z` under the
    /// [`crate::gravity::yaw_of`] definition; `variance` is its variance in
    /// rad^2. This is the only thing that ever reduces L1's yaw uncertainty,
    /// because gravity cannot (spec.md §4 L1: *"yaw drifts slowly and is
    /// arrested by L3"*).
    ///
    /// The measurement Jacobian is [`crate::gravity::yaw_jacobian_body`], which
    /// reads exactly 1 along the direction the gravity update refuses to touch.
    /// The two updates are therefore complementary by construction rather than
    /// by tuning: gravity moves the estimate about horizontal world axes only,
    /// this moves it about the vertical one. Ignored before initialisation, and
    /// for a non-positive or non-finite variance.
    pub fn correct_yaw(&mut self, yaw_world: Scalar, variance: Scalar) {
        if !self.initialized {
            return;
        }
        if !yaw_world.is_finite() || !variance.is_finite() || variance <= 0.0 {
            self.stats.rejected_samples += 1;
            return;
        }

        let mut h = Vec6::zeros();
        h.fixed_rows_mut::<3>(0)
            .copy_from(&yaw_jacobian_body(&self.attitude));
        let residual = wrap_angle(yaw_world - self.yaw());

        let p = self.covariance;
        let ph = p * h;
        let s = h.dot(&ph) + variance;
        if !s.is_finite() || s <= 0.0 {
            self.stats.numerical_failures += 1;
            return;
        }
        let k = ph / s;
        let ikh = Mat6::identity() - k * h.transpose();
        self.covariance = symmetrize(&(ikh * p * ikh.transpose() + k * k.transpose() * variance));
        self.inject(&(k * residual));
        self.stats.yaw_corrections += 1;
    }

    /// Attitude interpolated to an exact instant, `R_world_body`.
    ///
    /// Returns `None` outside the retained history window.
    ///
    /// Prefer this over [`OrientationFilter::attitude`] whenever the caller has
    /// a specific time in mind — a camera frame, say. `attitude()` is the state
    /// after the most recent *sample*, which at 200 Hz sits up to 5 ms from the
    /// frame it is about to be paired with. Differencing two such values gives
    /// an inter-frame rotation with up to 10 ms of timing error in it, and at
    /// drone angular rates that measured 0.9 degrees rms — enough to displace a
    /// feature by 9 px at the image edge and make the prediction worse than no
    /// prediction at all.
    #[must_use]
    pub fn attitude_at(&self, t: Timestamp) -> Option<So3> {
        self.history.at(t)
    }

    /// Relative rotation between two buffered times, for tracking prediction.
    ///
    /// Returns `delta` with `attitude(to) = attitude(from) · delta`, i.e. the
    /// rotation taking the body frame at `from` onto the body frame at `to`.
    /// Interpolates geodesically between samples; `None` when either time lies
    /// outside the retained window, which is bounded at
    /// [`crate::HISTORY_CAPACITY`] entries.
    #[must_use]
    pub fn delta_rotation(&self, from: Timestamp, to: Timestamp) -> Option<So3> {
        let a = self.history.at(from)?;
        let b = self.history.at(to)?;
        Some(a.inverse().compose(&b))
    }

    /// Oldest and newest times answerable by [`OrientationFilter::delta_rotation`].
    #[must_use]
    pub fn history_span(&self) -> Option<(Timestamp, Timestamp)> {
        self.history.span()
    }

    // -- internals ---------------------------------------------------------

    /// Set the attitude from one accelerometer sample, if it looks like gravity.
    fn try_initialize(&mut self, sample: &ImuSample) -> bool {
        let magnitude = sample.accel.norm();
        if !magnitude.is_finite()
            || magnitude < 1e-6
            || (magnitude - GRAVITY).abs() > self.config.gravity_gate
        {
            self.stats.gravity_rejected += 1;
            return false;
        }

        // R must satisfy R^T e_z = up_body, i.e. R maps the measured up
        // direction onto world up. The minimal such rotation is yaw-free, so the
        // session starts at yaw 0 by construction — an arbitrary but declared
        // heading origin, which is all L1 can offer.
        let up_body = sample.accel / magnitude;
        self.attitude = rotation_aligning(&up_body, &world_up());
        self.bias = Vec3::zeros();
        self.covariance = self.initial_covariance();
        self.initialized = true;
        self.since_renormalize = 0;
        self.stats.gravity_accepted += 1;
        true
    }

    /// Covariance immediately after initialisation: tilt known to one
    /// accelerometer sample, yaw not known at all.
    fn initial_covariance(&self) -> Mat6 {
        let v = yaw_axis_body(&self.attitude);
        let yaw_projector = v * v.transpose();
        let tilt_sigma = self.config.accel_noise / GRAVITY;
        let attitude = (Mat3::identity() - yaw_projector) * (tilt_sigma * tilt_sigma)
            + yaw_projector * INITIAL_YAW_VARIANCE;

        let mut p = Mat6::zeros();
        p.fixed_view_mut::<3, 3>(0, 0).copy_from(&attitude);
        p.fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&(Mat3::identity() * BIAS_PRIOR_VARIANCE));
        p
    }

    /// Integrate the gyro and propagate the covariance over one interval.
    fn propagate(&mut self, previous: &ImuSample, sample: &ImuSample, dt: Scalar) {
        // Trapezoidal rather than zero-order hold on the newest sample: holding
        // biases the integral by half a sample of angular acceleration, which at
        // 100 Hz and hand-held rates is larger than the gyro noise it sits on,
        // and it accumulates rather than averaging out.
        let rate = (previous.gyro + sample.gyro) * 0.5 - self.bias;
        let phi = rate * dt;
        let delta = So3::exp(&phi);
        let (f, mut q) = transition(&phi, dt, &self.config);

        if dt > MAX_TRUSTED_DT {
            // A stall in the event stream — a backgrounded tab, a dropped block
            // of DeviceMotion events. One sample cannot describe what the device
            // did across the gap, so the constant-rate assumption is worth
            // roughly nothing: model the integrated angle as uniform on
            // [0, |omega| dt], whose variance is (|omega| dt)^2 / 12.
            let gap = phi.norm_squared() / 12.0;
            let mut block = q.fixed_view_mut::<3, 3>(0, 0);
            block += Mat3::identity() * gap;
            log::debug!("orientation: {dt:.3} s gap in the motion stream");
        }

        self.attitude = self.attitude.compose(&delta);
        self.covariance = symmetrize(&(f * self.covariance * f.transpose() + q));
        self.since_renormalize += 1;
    }

    /// Correct roll and pitch against the measured gravity direction.
    fn update_gravity(&mut self, sample: &ImuSample) {
        let magnitude = sample.accel.norm();
        if !magnitude.is_finite()
            || magnitude < 1e-6
            || (magnitude - GRAVITY).abs() > self.config.gravity_gate
        {
            self.stats.gravity_rejected += 1;
            return;
        }

        let v = yaw_axis_body(&self.attitude);
        let residual = sample.accel / magnitude - v;

        // H = [hat(v), 0]. A rotation error along v IS a world yaw error, and
        // hat(v) v = 0, so yaw sits in the null space of the Jacobian by
        // construction. Nothing downstream has to remember to protect it.
        let mut h = Matrix3x6::zeros();
        h.fixed_view_mut::<3, 3>(0, 0).copy_from(&hat(&v));

        // Both sides are unit vectors, so the measurement noise is an angle: an
        // accelerometer error of `accel_noise` tilts a vector of length g by
        // accel_noise/g radians. The component of the residual along v is
        // second-order small and is annihilated by hat(v) regardless.
        //
        // But `accel_noise` describes the *sensor*, and the sensor is not the
        // dominant error here. The accelerometer measures specific force, and
        // whenever the device accelerates, the difference between that and
        // gravity is an unmodelled measurement error far larger than the noise
        // floor. The magnitude gate above bounds it only loosely: a body
        // accelerating horizontally at 2 m/s^2 reads |a| = 10.01, which passes a
        // 0.5 m/s^2 gate while pointing 11.5 degrees away from true down.
        //
        // Treating that as a 0.15 m/s^2 measurement makes the filter wildly
        // overconfident and lets every accelerating sample drag the attitude.
        // Measured on EuRoC MH_01, that produced 0.9 degrees rms of *inter-frame*
        // rotation error while the long-run tilt stayed at 2.5 degrees — the
        // signature of a per-sample tug that averages out but is noisy frame to
        // frame. The prediction built from it was worse than no prediction, and
        // sensor tier 2 lost four times as many frames as tier 1.
        //
        // So inflate the noise by the observed deviation from gravity, which is
        // the only in-band evidence we have of how much linear acceleration is
        // contaminating this sample. It is a lower bound — acceleration
        // perpendicular to gravity barely changes the magnitude — but it is
        // unbiased in the right direction and costs one square root.
        let deviation = (magnitude - GRAVITY).abs();
        let sigma = (self.config.accel_noise.powi(2) + deviation.powi(2)).sqrt() / GRAVITY;
        let r = Mat3::identity() * (sigma * sigma);

        // Deliberately no chi-squared gate on the innovation. A large residual
        // here is ambiguous between "the accelerometer is lying" and "the
        // attitude estimate is wrong", and rejecting it would make a large
        // initial attitude error permanent. Magnitude is the discriminator that
        // separates the two cases: linear acceleration changes |a|, an attitude
        // error does not.
        if self.apply_vector_update(&h, &residual, &r, true) {
            self.stats.gravity_accepted += 1;
        }
    }

    /// Observe the gyro bias directly while the device is not turning.
    fn update_zero_rate(&mut self, sample: &ImuSample, dt: Scalar) {
        let threshold = self.config.static_threshold;
        if threshold <= 0.0 {
            return;
        }
        // Decide on the *averaged* rate over the preceding window, never on the
        // sample about to be used as the measurement. One sample's noise is as
        // large as the whole threshold (see `crate::rates`), so a per-sample
        // test admits only the samples whose noise cancelled the bias and the
        // filter then estimates the bias as a fraction of its true value.
        let Some((mean_rate, window_seconds)) = self.rates.mean(sample.timestamp) else {
            return;
        };
        // Test the *measured* rate, not the de-biased one: a de-biased test lets
        // a wrong bias estimate hide a real rotation and then lock itself in.
        // The allowance carries the two uncertainties in that comparison — what
        // the bias might be, and what the window mean's own noise might be — so
        // that neither a large turn-on bias nor a short window gates out the
        // very update that would observe it.
        let bias_sigma = self
            .bias_covariance()
            .diagonal()
            .iter()
            .fold(0.0_f64, |a, &b| a.max(b))
            .max(0.0)
            .sqrt();
        let mean_sigma = self.config.gyro_noise / window_seconds.max(1e-6).sqrt();
        let allowance = 3.0 * (bias_sigma * bias_sigma + mean_sigma * mean_sigma).sqrt();
        if mean_rate.norm() > threshold + allowance {
            return;
        }

        let mut h = Matrix3x6::zeros();
        h.fixed_view_mut::<3, 3>(0, 3).copy_from(&Mat3::identity());
        let residual = sample.gyro - self.bias;

        // "Static" only means "below the threshold", so the pseudo-measurement
        // carries that slack as noise (three sigma = the threshold) on top of
        // the gyro's own white noise over the interval.
        let slack = threshold / 3.0;
        let white = self.config.gyro_noise / dt.max(1e-6).sqrt();
        let r = Mat3::identity() * (slack * slack + white * white);

        if self.apply_vector_update(&h, &residual, &r, false) {
            self.stats.static_updates += 1;
        }
    }

    /// Joseph-form Kalman update for a 3-vector measurement.
    ///
    /// `protect_yaw` projects the attitude rows of the gain off the yaw
    /// direction. With an isotropic covariance the projection is a no-op — the
    /// null space of `hat(v)` already guarantees it — but once the covariance
    /// picks up cross-axis correlations, linearisation error leaks a little yaw
    /// correction out of a measurement that carries no yaw information. That
    /// leak shows up downstream as an L6 yaw variance which shrinks although no
    /// heading was ever observed, which is exactly the overconfidence spec.md §6
    /// L6 calls worse than no covariance at all. Constraining the unobservable
    /// direction explicitly is the observability-constrained EKF of Huang,
    /// Mourikis & Roumeliotis, in its simplest possible form.
    ///
    /// The Joseph form is not decoration here: `(I - KH)P` is only valid for the
    /// optimal gain, and the projection above makes the gain sub-optimal on
    /// purpose.
    fn apply_vector_update(
        &mut self,
        h: &Matrix3x6<Scalar>,
        residual: &Vec3,
        r: &Mat3,
        protect_yaw: bool,
    ) -> bool {
        let p = self.covariance;
        let pht = p * h.transpose();
        let s = h * pht + r;
        let Some(s_inv) = s.try_inverse() else {
            self.stats.numerical_failures += 1;
            return false;
        };
        let mut k = pht * s_inv;

        if protect_yaw {
            let v = yaw_axis_body(&self.attitude);
            let projected =
                (Mat3::identity() - v * v.transpose()) * k.fixed_view::<3, 3>(0, 0).into_owned();
            k.fixed_view_mut::<3, 3>(0, 0).copy_from(&projected);
        }

        let ikh = Mat6::identity() - k * h;
        let next = symmetrize(&(ikh * p * ikh.transpose() + k * r * k.transpose()));
        if next.iter().any(|v| !v.is_finite()) {
            self.stats.numerical_failures += 1;
            return false;
        }
        self.covariance = next;
        self.inject(&(k * residual));
        true
    }

    /// Fold the error-state mean into the nominal state and reset it to zero.
    ///
    /// Injecting `δθ` replaces the body frame the covariance is written in:
    /// afterwards `R' = R · exp(δθ)`, so the new body axes are the old ones
    /// turned by `exp(δθ)` and the attitude block has to be re-expressed in
    /// them, `P_θθ ← E^T P_θθ E` with `E = exp(δθ)` (and `P_θb ← E^T P_θb`).
    /// This is a change of coordinates, not a refinement, and skipping it is
    /// not a small error here: the yaw direction carries `pi^2/3` rad^2, so
    /// leaving `P` in the stale frame tips a fraction `sin^2|δθ|` of that into
    /// roll and pitch on **every** correction. That inflates the gain, which
    /// enlarges the next `|δθ|`, which leaks more — a positive feedback that
    /// measurably multiplies the static roll/pitch error (see
    /// `tests/scenarios.rs::static_level_converges_to_gravity_and_stays_there`).
    ///
    /// The full error-state reset of Solà, *Quaternion kinematics for the
    /// error-state KF* §6 is `P_θθ ← Jr(δθ) P_θθ Jr(δθ)^T`, which factors as
    /// this frame change followed by `Jl(δθ) · Jl(δθ)^T` — a further rotation
    /// of `P` by `|δθ|/2`. That second half is a first-order expansion in the
    /// error, and it is only valid while the error is small. Along yaw the
    /// error is *not* small (one sigma is over a radian), so the expansion
    /// produces exactly the spurious leak described above with a quarter of the
    /// magnitude. It is therefore deliberately omitted: dropping it is
    /// second-order wherever the linearisation holds, and it makes the
    /// observability statement exact — the variance about the world vertical is
    /// invariant under a gravity correction, not merely nearly so.
    fn inject(&mut self, dx: &Vec6) {
        if !dx.iter().all(|v| v.is_finite()) {
            self.stats.numerical_failures += 1;
            return;
        }
        let dtheta = Vec3::new(dx[0], dx[1], dx[2]);
        let dbias = Vec3::new(dx[3], dx[4], dx[5]);
        self.attitude = self.attitude.plus(&dtheta);
        self.bias += dbias;

        let e = So3::exp(&dtheta).matrix();
        let att = self.covariance.fixed_view::<3, 3>(0, 0).into_owned();
        let cross = self.covariance.fixed_view::<3, 3>(0, 3).into_owned();
        self.covariance
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(e.transpose() * att * e));
        self.covariance
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(e.transpose() * cross));
        self.covariance
            .fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&(cross.transpose() * e));
    }

    fn renormalize_if_due(&mut self) {
        if self.since_renormalize >= RENORMALIZE_INTERVAL {
            self.attitude = self.attitude.normalized();
            self.since_renormalize = 0;
        }
    }
}

/// State transition and process noise for one propagation interval.
///
/// With `R_true = R · exp(δθ)`, `b_true = b + δb` and `phi = (omega_m - b) dt`:
///
/// ```text
/// F = [ Exp(phi)^T   -Jr(phi) dt ]      Q = [ sigma_g^2 dt Jr Jr^T        0        ]
///     [     0             I      ]          [          0          sigma_bw^2 dt I  ]
/// ```
///
/// The `-Jr(phi) dt` block is what makes the bias estimable: it is the only
/// path by which a bias error becomes an attitude error the accelerometer can
/// see. Gyro white noise enters through the same coefficient as the bias error,
/// which is why `Q`'s attitude block carries `Jr Jr^T` rather than `I`.
fn transition(phi: &Vec3, dt: Scalar, config: &OrientationConfig) -> (Mat6, Mat6) {
    let jr = So3::right_jacobian(phi);
    let mut f = Mat6::identity();
    f.fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&So3::exp(phi).matrix().transpose());
    f.fixed_view_mut::<3, 3>(0, 3).copy_from(&(-jr * dt));

    let mut q = Mat6::zeros();
    q.fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(jr * jr.transpose() * (config.gyro_noise * config.gyro_noise * dt)));
    q.fixed_view_mut::<3, 3>(3, 3)
        .copy_from(&(Mat3::identity() * (config.gyro_bias_walk * config.gyro_bias_walk * dt)));
    (f, q)
}

fn is_finite(v: &Vec3) -> bool {
    v.iter().all(|x| x.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use wslam_core::covariance::is_valid_covariance;
    use wslam_core::DeterministicRng;

    fn level_sample(t: f64) -> ImuSample {
        ImuSample::new(
            Timestamp::from_seconds(t),
            Vec3::zeros(),
            Vec3::new(0.0, 0.0, GRAVITY),
        )
    }

    #[test]
    fn transition_jacobian_matches_central_differences() {
        // Differentiate the *nonlinear* error propagation and compare to F.
        // A tautological version of this test would rebuild F from the same
        // formula; this one re-derives it from group operations only.
        let config = OrientationConfig::default();
        let dt = 0.011;
        let measured = Vec3::new(0.7, -0.4, 1.1);
        let bias = Vec3::new(0.01, -0.02, 0.005);
        let r = So3::exp(&Vec3::new(0.3, -0.2, 0.5));

        let phi = (measured - bias) * dt;
        let (f, _) = transition(&phi, dt, &config);
        let nominal_next = r.compose(&So3::exp(&phi));

        let propagate_error = |e: Vec6| -> Vec6 {
            let dtheta = Vec3::new(e[0], e[1], e[2]);
            let dbias = Vec3::new(e[3], e[4], e[5]);
            let next_true = r
                .plus(&dtheta)
                .compose(&So3::exp(&((measured - (bias + dbias)) * dt)));
            let out_theta = next_true.minus(&nominal_next);
            Vec6::new(
                out_theta.x,
                out_theta.y,
                out_theta.z,
                dbias.x,
                dbias.y,
                dbias.z,
            )
        };

        let h = 1e-6;
        for col in 0..6 {
            let mut plus = Vec6::zeros();
            plus[col] = h;
            let numerical = (propagate_error(plus) - propagate_error(-plus)) / (2.0 * h);
            for row in 0..6 {
                assert_relative_eq!(numerical[row], f[(row, col)], epsilon = 1e-7);
            }
        }
    }

    #[test]
    fn gravity_jacobian_matches_central_differences_and_annihilates_yaw() {
        let r = So3::exp(&Vec3::new(0.2, 0.9, -0.4));
        let v = yaw_axis_body(&r);
        let jacobian = hat(&v);
        let predict = |d: Vec3| yaw_axis_body(&r.plus(&d));

        let h = 1e-6;
        for col in 0..3 {
            let mut d = Vec3::zeros();
            d[col] = h;
            let numerical = (predict(d) - predict(-d)) / (2.0 * h);
            for row in 0..3 {
                assert_relative_eq!(numerical[row], jacobian[(row, col)], epsilon = 1e-8);
            }
        }
        // The yaw direction produces no measurement change, exactly.
        assert_relative_eq!(jacobian * v, Vec3::zeros(), epsilon = 1e-15);
    }

    #[test]
    fn process_noise_is_psd_and_scales_with_the_interval() {
        let config = OrientationConfig::default();
        let phi = Vec3::new(0.02, -0.01, 0.03);
        let (_, q1) = transition(&phi, 0.01, &config);
        let (_, q2) = transition(&phi, 0.02, &config);
        assert!(is_valid_covariance(&q1, 1e-12));
        assert_relative_eq!(q2[(0, 0)], 2.0 * q1[(0, 0)], epsilon = 1e-18);
        assert_relative_eq!(q2[(3, 3)], 2.0 * q1[(3, 3)], epsilon = 1e-18);
        assert!(q1.symmetric_eigen().eigenvalues.iter().all(|&e| e >= 0.0));
    }

    /// Variance about the **world** vertical, `e_z^T (R P R^T) e_z`, which in
    /// body coordinates is `v^T P v` with `v = yaw_axis_body(R)`.
    ///
    /// The direction has to be recomputed from the current attitude every time:
    /// `P` is written in body axes, and a correction turns those axes, so the
    /// body vector that points along the world vertical is not the same vector
    /// before and after. This is the quantity the whole crate calls "the yaw
    /// variance", and it is the one that must not shrink without a heading
    /// observation.
    fn world_yaw_variance(f: &OrientationFilter) -> Scalar {
        let v = yaw_axis_body(&f.attitude);
        (v.transpose() * f.covariance() * v)[(0, 0)]
    }

    #[test]
    fn a_gravity_update_leaves_the_yaw_variance_exactly_unchanged() {
        // The structural claim, tested against a deliberately correlated,
        // anisotropic covariance — the case where a naive gain leaks yaw.
        let mut f = OrientationFilter::new(OrientationConfig::default());
        f.integrate(&level_sample(0.0));
        assert!(f.is_initialized());

        // Poke in cross-correlated garbage that is still a valid covariance.
        let mut rng = DeterministicRng::new("yaw-invariant", 4242);
        let a = Mat6::from_fn(|_, _| rng.normal());
        f.covariance = symmetrize(&(a * a.transpose())) + Mat6::identity() * 1e-6;
        f.attitude = So3::exp(&Vec3::new(0.3, -0.5, 0.9));

        let v = yaw_axis_body(&f.attitude);
        let before = world_yaw_variance(&f);

        // A measurement consistent with a 5 degree tilt error.
        let tilted = So3::exp(&Vec3::new(0.05, 0.03, 0.0));
        let accel = tilted.act(&(v * GRAVITY));
        f.update_gravity(&ImuSample::new(
            Timestamp::from_seconds(0.01),
            Vec3::zeros(),
            accel,
        ));
        assert_eq!(f.stats().gravity_accepted, 2);
        // The correction is large enough that the body frame really did turn,
        // so this is a genuine test of the frame change in `inject` and not of
        // an identity transform.
        assert!(yaw_axis_body(&f.attitude).dot(&v) < 1.0 - 1e-6);

        let after = world_yaw_variance(&f);
        assert_relative_eq!(after, before, max_relative = 1e-9);
        assert!(is_valid_covariance(&f.covariance, 1e-9));
    }

    #[test]
    fn the_frame_change_on_injection_is_what_keeps_the_yaw_variance_put() {
        // Companion to the test above: without re-expressing P in the corrected
        // body axes, the pi^2/3 rad^2 sitting on the yaw direction spills into
        // roll and pitch, and the reported world-vertical variance falls. This
        // pins the mechanism, not just the outcome.
        let mut f = OrientationFilter::new(OrientationConfig::default());
        f.integrate(&level_sample(0.0));
        f.attitude = So3::exp(&Vec3::new(0.3, -0.5, 0.9));
        f.covariance = f.initial_covariance();

        let before = world_yaw_variance(&f);
        // `initial_covariance` is `Y·vv^T + T·(I - vv^T)`, so the variance about
        // any direction at angle a from v is `Y cos^2 a + T sin^2 a`.
        let tilt_var = (OrientationConfig::default().accel_noise / GRAVITY).powi(2);

        // Inject a 3 degree tilt correction, once with the frame change and
        // once without.
        let dtheta = yaw_axis_body(&f.attitude).cross(&Vec3::x()).normalize() * 0.05;
        let mut stale = f.clone();
        stale.attitude = stale.attitude.plus(&dtheta); // the old, frame-less inject
        let mut correct = f.clone();
        let mut dx = Vec6::zeros();
        dx.fixed_rows_mut::<3>(0).copy_from(&dtheta);
        correct.inject(&dx);

        assert_relative_eq!(world_yaw_variance(&correct), before, max_relative = 1e-12);
        let leaked = before - world_yaw_variance(&stale);
        // (pi^2/3 - T)·sin^2(0.05) = 8.2e-3 rad^2 of heading uncertainty
        // reclassified as tilt uncertainty, against a tilt variance of 2.3e-4:
        // one 3-degree correction misreports 35 times the whole tilt variance.
        assert_relative_eq!(
            leaked,
            (before - tilt_var) * 0.05_f64.sin().powi(2),
            max_relative = 1e-9
        );
        assert!(leaked > 30.0 * tilt_var);
    }

    #[test]
    fn without_the_projection_a_correlated_covariance_does_leak_yaw() {
        // Proves the projection is load-bearing rather than a no-op: the same
        // update with protect_yaw = false shrinks the yaw variance.
        let mut f = OrientationFilter::new(OrientationConfig::default());
        f.integrate(&level_sample(0.0));
        let mut rng = DeterministicRng::new("yaw-leak", 99);
        let a = Mat6::from_fn(|_, _| rng.normal());
        f.covariance = symmetrize(&(a * a.transpose())) + Mat6::identity() * 1e-6;
        f.attitude = So3::exp(&Vec3::new(0.3, -0.5, 0.9));

        let v = yaw_axis_body(&f.attitude);
        let mut e = Vec6::zeros();
        e.fixed_rows_mut::<3>(0).copy_from(&v);
        let before = (e.transpose() * f.covariance * e)[(0, 0)];

        let mut h = Matrix3x6::zeros();
        h.fixed_view_mut::<3, 3>(0, 0).copy_from(&hat(&v));
        let sigma = f.config.accel_noise / GRAVITY;
        let r = Mat3::identity() * (sigma * sigma);
        f.apply_vector_update(&h, &Vec3::new(0.05, 0.03, 0.0), &r, false);

        let after = (e.transpose() * f.covariance * e)[(0, 0)];
        assert!(
            after < before * 0.999,
            "unprotected update should have leaked yaw: {before} -> {after}"
        );
    }

    #[test]
    fn initialisation_waits_for_a_sample_that_looks_like_gravity() {
        let mut f = OrientationFilter::new(OrientationConfig::default());
        // Under a 3 g shove the accelerometer is not measuring gravity.
        f.integrate(&ImuSample::new(
            Timestamp::from_seconds(0.0),
            Vec3::zeros(),
            Vec3::new(0.0, 0.0, 3.0 * GRAVITY),
        ));
        assert!(!f.is_initialized());
        assert!(f.delta_rotation(Timestamp::ZERO, Timestamp::ZERO).is_none());

        f.integrate(&level_sample(0.01));
        assert!(f.is_initialized());
        assert_relative_eq!(f.attitude().matrix(), Mat3::identity(), epsilon = 1e-12);
    }

    #[test]
    fn initialisation_recovers_a_known_tilt_and_starts_at_zero_yaw() {
        let truth = So3::exp(&Vec3::new(0.4, -0.25, 0.0));
        let accel = truth.inverse().act(&(Vec3::z() * GRAVITY));
        let mut f = OrientationFilter::new(OrientationConfig::default());
        f.integrate(&ImuSample::new(Timestamp::ZERO, Vec3::zeros(), accel));

        assert!(f.is_initialized());
        assert_relative_eq!(
            crate::gravity::tilt_between(&f.attitude(), &truth),
            0.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(f.yaw(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn initialisation_survives_an_upside_down_device() {
        // The antiparallel degenerate case: gravity reads along -z in body.
        let mut f = OrientationFilter::new(OrientationConfig::default());
        f.integrate(&ImuSample::new(
            Timestamp::ZERO,
            Vec3::zeros(),
            Vec3::new(0.0, 0.0, -GRAVITY),
        ));
        assert!(f.is_initialized());
        // Body -z must now point along world up.
        assert_relative_eq!(f.gravity_body(), Vec3::z(), epsilon = 1e-9);
    }

    #[test]
    fn initial_covariance_knows_tilt_and_admits_it_knows_no_yaw() {
        let mut f = OrientationFilter::new(OrientationConfig::default());
        f.integrate(&level_sample(0.0));
        let p = f.covariance();
        let tilt_sigma = OrientationConfig::default().accel_noise / GRAVITY;
        assert_relative_eq!(p[(0, 0)], tilt_sigma * tilt_sigma, epsilon = 1e-15);
        assert_relative_eq!(p[(1, 1)], tilt_sigma * tilt_sigma, epsilon = 1e-15);
        assert_relative_eq!(p[(2, 2)], INITIAL_YAW_VARIANCE, epsilon = 1e-12);
        assert!(is_valid_covariance(&f.full_covariance(), 1e-12));
    }

    #[test]
    fn non_finite_duplicate_and_out_of_order_samples_are_refused() {
        let mut f = OrientationFilter::new(OrientationConfig::default());
        f.integrate(&level_sample(0.0));
        f.integrate(&level_sample(0.01));
        let accepted = f.stats().samples;

        f.integrate(&ImuSample::new(
            Timestamp::from_seconds(0.02),
            Vec3::new(f64::NAN, 0.0, 0.0),
            Vec3::new(0.0, 0.0, GRAVITY),
        ));
        f.integrate(&level_sample(0.01)); // duplicate
        f.integrate(&level_sample(0.005)); // out of order

        assert_eq!(f.stats().samples, accepted);
        assert_eq!(f.stats().rejected_samples, 3);
        assert_eq!(f.last_timestamp(), Some(Timestamp::from_seconds(0.01)));
    }

    #[test]
    fn a_stream_gap_inflates_the_attitude_covariance() {
        let mut f = OrientationFilter::new(OrientationConfig::default());
        f.integrate(&ImuSample::new(
            Timestamp::ZERO,
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, GRAVITY),
        ));
        // Force the gate closed so only propagation is compared.
        let shoved = Vec3::new(0.0, 0.0, 3.0 * GRAVITY);
        let mut small = f.clone();
        small.integrate(&ImuSample::new(
            Timestamp::from_seconds(0.01),
            Vec3::new(0.0, 0.0, 1.0),
            shoved,
        ));
        let mut gapped = f.clone();
        gapped.integrate(&ImuSample::new(
            Timestamp::from_seconds(1.0),
            Vec3::new(0.0, 0.0, 1.0),
            shoved,
        ));
        let tilt = |x: &OrientationFilter| x.covariance()[(0, 0)];
        assert!(
            tilt(&gapped) > 100.0 * tilt(&small),
            "{} vs {}",
            tilt(&gapped),
            tilt(&small)
        );
    }

    #[test]
    fn the_gate_rejects_what_it_should_and_the_ablation_accepts_everything() {
        let shaken = ImuSample::new(
            Timestamp::from_seconds(0.01),
            Vec3::zeros(),
            Vec3::new(0.0, 0.0, GRAVITY + 4.0),
        );
        let mut gated = OrientationFilter::new(OrientationConfig::default());
        gated.integrate(&level_sample(0.0));
        gated.integrate(&shaken);
        assert_eq!(gated.stats().gravity_accepted, 1); // init only
        assert_eq!(gated.stats().gravity_rejected, 1);

        let mut ungated = OrientationFilter::new(OrientationConfig::ungated());
        ungated.integrate(&level_sample(0.0));
        ungated.integrate(&shaken);
        assert_eq!(ungated.stats().gravity_accepted, 2);
        assert_eq!(ungated.stats().gravity_rejected, 0);
    }

    #[test]
    fn correct_yaw_is_ignored_before_initialisation_and_for_bad_variance() {
        let mut f = OrientationFilter::new(OrientationConfig::default());
        f.correct_yaw(1.0, 0.01);
        assert_eq!(f.stats().yaw_corrections, 0);

        f.integrate(&level_sample(0.0));
        f.correct_yaw(1.0, 0.0);
        f.correct_yaw(1.0, -1.0);
        f.correct_yaw(f64::NAN, 0.01);
        assert_eq!(f.stats().yaw_corrections, 0);
        assert_relative_eq!(f.yaw(), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn correct_yaw_moves_yaw_and_leaves_tilt_alone() {
        let mut f = OrientationFilter::new(OrientationConfig::default());
        let truth = So3::exp(&Vec3::new(0.3, -0.2, 0.0));
        f.integrate(&ImuSample::new(
            Timestamp::ZERO,
            Vec3::zeros(),
            truth.inverse().act(&(Vec3::z() * GRAVITY)),
        ));
        let tilt_before = crate::gravity::tilt_between(&f.attitude(), &truth);

        // Yaw variance starts at pi^2/3, so a 1 degree observation dominates.
        f.correct_yaw(0.7, 3.0e-4);
        assert_eq!(f.stats().yaw_corrections, 1);
        assert_relative_eq!(f.yaw(), 0.7, epsilon = 1e-3);
        // The heading correction spills into tilt in proportion to the
        // tilt/yaw variance ratio, which is ~1e-4 here. Bounded, and four
        // orders of magnitude below the 0.7 rad of yaw that was corrected.
        assert_relative_eq!(
            crate::gravity::tilt_between(&f.attitude(), &truth),
            tilt_before,
            epsilon = 1e-4
        );
        assert!(f.covariance()[(2, 2)] < 1e-3, "yaw variance must collapse");
    }

    #[test]
    fn correct_yaw_takes_the_short_way_round_the_circle() {
        let mut f = OrientationFilter::new(OrientationConfig::default());
        f.integrate(&level_sample(0.0));
        f.correct_yaw(3.0, 1e-6);
        assert_relative_eq!(f.yaw(), 3.0, epsilon = 1e-3);

        // Observing -3.0 from +3.0 is a +0.283 rad correction across the branch
        // cut, not a -6.0 rad one the long way round.
        let before = f.yaw();
        f.correct_yaw(-3.0, 1e-6);
        let step = wrap_angle(f.yaw() - before);
        assert!(step > 0.0 && step < 0.3, "{before} -> {} ({step})", f.yaw());
        // Repeated observations close the remaining gap across the branch cut.
        let mut error = wrap_angle(f.yaw() + 3.0).abs();
        for _ in 0..200 {
            f.correct_yaw(-3.0, 1e-6);
            let next = wrap_angle(f.yaw() + 3.0).abs();
            assert!(next <= error + 1e-12, "{error} -> {next}");
            error = next;
        }
        assert!(error < 5e-3, "residual yaw error {error}");
    }

    #[test]
    fn a_gravity_correction_is_a_rotation_about_a_horizontal_world_axis() {
        // The exact structural statement of "gravity does not touch heading",
        // and the one that holds at every attitude: the correction's axis, in
        // WORLD coordinates, has no vertical component.
        let mut f = OrientationFilter::new(OrientationConfig::default());
        let truth = So3::exp(&Vec3::new(0.5, -0.3, 1.1));
        f.integrate(&ImuSample::new(
            Timestamp::ZERO,
            Vec3::zeros(),
            truth.inverse().act(&(Vec3::z() * GRAVITY)),
        ));
        // Rough up the covariance so the gain has cross-axis structure.
        let mut rng = DeterministicRng::new("horizontal-axis", 17);
        let a = Mat6::from_fn(|_, _| rng.normal());
        f.covariance = symmetrize(&(a * a.transpose())) + Mat6::identity() * 1e-6;

        let before = f.attitude();
        let wrong = So3::exp(&Vec3::new(0.06, -0.04, 0.02));
        let accel = wrong.act(&(yaw_axis_body(&before) * GRAVITY));
        f.update_gravity(&ImuSample::new(
            Timestamp::from_seconds(0.01),
            Vec3::zeros(),
            accel,
        ));

        let body_axis = f.attitude().minus(&before);
        assert!(
            body_axis.norm() > 1e-4,
            "the update must have done something"
        );
        let world_axis = before.act(&body_axis);
        assert_relative_eq!(world_axis.z / world_axis.norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn reset_returns_to_the_uninitialised_state() {
        let mut f = OrientationFilter::new(OrientationConfig::default());
        f.integrate(&level_sample(0.0));
        f.integrate(&level_sample(0.01));
        f.reset();
        assert!(!f.is_initialized());
        assert_eq!(f.stats(), FilterStats::default());
        assert!(f.history_span().is_none());
        assert_eq!(f.gyro_bias(), Vec3::zeros());
    }

    #[test]
    fn delta_rotation_recovers_a_known_relative_rotation() {
        let mut f = OrientationFilter::new(OrientationConfig::default());
        let rate = Vec3::new(0.0, 0.0, 0.5);
        for i in 0..=100 {
            let t = i as f64 * 0.01;
            f.integrate(&ImuSample::new(
                Timestamp::from_seconds(t),
                rate,
                Vec3::new(0.0, 0.0, GRAVITY),
            ));
        }
        // Between 0.2 s and 0.7 s the device turned 0.5 * 0.5 = 0.25 rad.
        let d = f
            .delta_rotation(Timestamp::from_seconds(0.2), Timestamp::from_seconds(0.7))
            .unwrap();
        assert_relative_eq!(d.log(), Vec3::z() * 0.25, epsilon = 1e-6);

        // And it composes, at sample times and between them.
        let (a, b, c) = (
            Timestamp::from_seconds(0.15),
            Timestamp::from_seconds(0.435),
            Timestamp::from_seconds(0.82),
        );
        let composed = f
            .delta_rotation(a, b)
            .unwrap()
            .compose(&f.delta_rotation(b, c).unwrap());
        assert_relative_eq!(
            composed.matrix(),
            f.delta_rotation(a, c).unwrap().matrix(),
            epsilon = 1e-9
        );
        // Reversing inverts.
        assert_relative_eq!(
            f.delta_rotation(c, a).unwrap().matrix(),
            f.delta_rotation(a, c).unwrap().inverse().matrix(),
            epsilon = 1e-9
        );
        assert_relative_eq!(
            f.delta_rotation(b, b).unwrap().angle(),
            0.0,
            epsilon = 1e-15
        );
    }

    #[test]
    fn delta_rotation_declines_outside_the_bounded_history() {
        let mut f = OrientationFilter::new(OrientationConfig::default());
        for i in 0..(HISTORY_CAPACITY + 500) {
            f.integrate(&level_sample(i as f64 * 0.01));
        }
        let (lo, hi) = f.history_span().unwrap();
        assert_eq!(f.history.len(), HISTORY_CAPACITY);
        assert!(f.delta_rotation(Timestamp::ZERO, hi).is_none());
        assert!(f.delta_rotation(lo, hi).is_some());
        assert!(f.delta_rotation(lo, hi.offset_seconds(0.5)).is_none());
    }
}
