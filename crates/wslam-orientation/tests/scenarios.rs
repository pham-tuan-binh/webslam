//! The spec.md §6 L1 measurements, run synthetically at Tier 1.
//!
//! > - Turntable at known rate -> integrated angle error over 60 s (deg).
//! > - Static on a level surface -> roll/pitch error vs gravity (deg).
//! > - Yaw drift rate (deg/min), vision disabled.
//!
//! Tier 1 is *"everything with a closed-form answer, on synthetic input"*, so
//! every truth here is written down in closed form and the filter is never
//! consulted for its own ground truth. The rig below is the only shared
//! machinery: it turns a known attitude trajectory into the samples a phone
//! would have delivered.

use wslam_core::imu::GRAVITY;
use wslam_core::math::{Scalar, So3, Vec3};
use wslam_core::{DeterministicRng, ImuSample, Timestamp};
use wslam_orientation::gravity::{tilt_between, wrap_angle};
use wslam_orientation::{OrientationConfig, OrientationFilter};

const HZ: Scalar = 100.0;
const DT: Scalar = 1.0 / HZ;

/// One sigma of turn-on gyro bias as the filter models it, rad/s.
///
/// Mirrors `filter::BIAS_PRIOR_VARIANCE`, which is private. Scenarios that need
/// an error the filter's covariance can legitimately own are built from this,
/// so that "the estimate is one sigma out" is a statement about the shipped
/// noise model rather than a number chosen to make a test pass.
const BIAS_SIGMA: Scalar = 1.732_050_807_568_877_2e-2; // sqrt(3e-4)

/// Budget for yaw leaked by a gravity correction, per squared degree of tilt.
///
/// The leak is second order in the tilt being removed (see
/// `the_yaw_leak_is_second_order_in_the_tilt_removed` for the measurement), so
/// the budget has to be quadratic too. Measured worst case across a 7-to-42
/// degree sweep is 2.5e-6/deg; this is 1e-5, about four times that, which
/// leaves room for platform floating-point differences without leaving room for
/// a real first-order regression — one would blow through it by orders of
/// magnitude.
const LEAK_COEFFICIENT: Scalar = 1e-5;

/// Synthetic IMU. Given the true attitude and the true motion, produce the
/// sample a device would have reported: gyro contaminated by a constant bias
/// and white noise, accelerometer reporting specific force in the body frame.
struct Rig {
    rng: DeterministicRng,
    /// Per-sample gyro standard deviation, rad/s. Equals the configured noise
    /// density divided by sqrt(dt), which is what makes the filter's process
    /// noise the correct model of this rig rather than a guess.
    gyro_sigma: Scalar,
    /// Per-sample accelerometer standard deviation, m/s^2.
    accel_sigma: Scalar,
    bias: Vec3,
}

impl Rig {
    /// A perfect IMU: no noise, no bias. The closed-form cases use this.
    fn clean(seed: u64) -> Self {
        Rig {
            rng: DeterministicRng::new("l1-rig", seed),
            gyro_sigma: 0.0,
            accel_sigma: 0.0,
            bias: Vec3::zeros(),
        }
    }

    /// A realistic phone IMU, matched to `config`.
    fn noisy(seed: u64, config: &OrientationConfig) -> Self {
        Rig {
            rng: DeterministicRng::new("l1-rig", seed),
            gyro_sigma: config.gyro_noise / DT.sqrt(),
            accel_sigma: config.accel_noise,
            bias: Vec3::zeros(),
        }
    }

    fn with_bias(mut self, bias: Vec3) -> Self {
        self.bias = bias;
        self
    }

    fn noise(&mut self, sigma: Scalar) -> Vec3 {
        if sigma == 0.0 {
            return Vec3::zeros();
        }
        Vec3::new(
            self.rng.normal() * sigma,
            self.rng.normal() * sigma,
            self.rng.normal() * sigma,
        )
    }

    /// One sample. `body_rate` is the true angular velocity in the body frame,
    /// `linear_world` the true translational acceleration in the world frame.
    fn sample(
        &mut self,
        t: Scalar,
        attitude: &So3,
        body_rate: Vec3,
        linear_world: Vec3,
    ) -> ImuSample {
        let gyro = body_rate + self.bias + self.noise(self.gyro_sigma);
        // Specific force: what the accelerometer actually reports is the
        // reaction to gravity plus the device's own acceleration.
        let specific_force = linear_world + Vec3::z() * GRAVITY;
        let accel = attitude.inverse().act(&specific_force) + self.noise(self.accel_sigma);
        ImuSample::new(Timestamp::from_seconds(t), gyro, accel)
    }
}

fn degrees(radians: Scalar) -> Scalar {
    radians.to_degrees()
}

/// Total angular error between an estimate and the truth, radians.
fn angle_error(estimate: &So3, truth: &So3) -> Scalar {
    estimate.minus(truth).norm()
}

/// Root-mean-square of a slice.
fn rms(values: &[Scalar]) -> Scalar {
    (values.iter().map(|v| v * v).sum::<Scalar>() / values.len() as Scalar).sqrt()
}

// ---------------------------------------------------------------------------
// Turntable: known rate, 60 s, integrated angle error.
// ---------------------------------------------------------------------------

#[test]
fn turntable_about_the_vertical_holds_to_a_tenth_of_a_degree_over_60s() {
    // A real turntable spins about the vertical, so gravity is constant and the
    // accelerometer updates run throughout. That makes this two tests in one:
    // the integrated angle must be right, AND 6000 gravity updates must not have
    // stolen any of it — which is the whole yaw-is-unobservable claim, measured.
    let rate = 0.513_7; // rad/s, deliberately not a round number
    let axis = Vec3::z();
    let mut rig = Rig::clean(1);
    let mut filter = OrientationFilter::new(OrientationConfig::default());

    let steps = (60.0 * HZ) as usize;
    for i in 0..=steps {
        let t = i as Scalar * DT;
        let truth = So3::exp(&(axis * (rate * t)));
        filter.integrate(&rig.sample(t, &truth, axis * rate, Vec3::zeros()));
    }

    let truth = So3::exp(&(axis * (rate * 60.0)));
    let error = degrees(angle_error(&filter.attitude(), &truth));
    assert!(
        error < 0.1,
        "integrated angle error {error:.5} deg over 60 s"
    );
    assert!(filter.stats().gravity_accepted > 6000);
    // The turntable never accelerates translationally, so nothing was gated out.
    assert_eq!(filter.stats().gravity_rejected, 0);
}

#[test]
fn pure_gyro_integration_about_an_arbitrary_axis_holds_over_60s() {
    // Gravity deliberately unavailable for the whole run (the device is being
    // shaken hard enough that every sample fails the gate), so this isolates the
    // integrator: 30 radians of rotation against a closed-form answer.
    let rate = 0.513_7;
    let axis = Vec3::new(0.3, -0.5, 0.81).normalize();
    let mut rig = Rig::clean(2);
    let mut filter = OrientationFilter::new(OrientationConfig::default());

    // One good sample to establish the (level) attitude, matching truth at t=0.
    filter.integrate(&rig.sample(0.0, &So3::identity(), axis * rate, Vec3::zeros()));
    assert!(filter.is_initialized());

    let steps = (60.0 * HZ) as usize;
    for i in 1..=steps {
        let t = i as Scalar * DT;
        let mut sample = rig.sample(t, &So3::identity(), axis * rate, Vec3::zeros());
        sample.accel = Vec3::new(0.0, 0.0, 20.0); // 2 g: far outside the gate
        filter.integrate(&sample);
    }

    let truth = So3::exp(&(axis * (rate * 60.0)));
    let error = degrees(angle_error(&filter.attitude(), &truth));
    assert!(
        error < 0.1,
        "integrated angle error {error:.6} deg over 60 s"
    );
    assert_eq!(filter.stats().gravity_accepted, 1, "init only");
    assert_eq!(filter.stats().gravity_rejected, steps as u64);
}

// ---------------------------------------------------------------------------
// Static on a level surface.
// ---------------------------------------------------------------------------

#[test]
fn static_level_converges_to_gravity_and_stays_there() {
    let config = OrientationConfig::default();
    let mut rig = Rig::noisy(3, &config);
    let mut filter = OrientationFilter::new(config);
    let truth = So3::identity();

    let mut early = Vec::new();
    let mut late = Vec::new();
    let steps = (30.0 * HZ) as usize;
    for i in 0..=steps {
        let t = i as Scalar * DT;
        filter.integrate(&rig.sample(t, &truth, Vec3::zeros(), Vec3::zeros()));
        let tilt = degrees(tilt_between(&filter.attitude(), &truth));
        if (5.0..10.0).contains(&t) {
            early.push(tilt);
        } else if t >= 25.0 {
            late.push(tilt);
        }
    }

    let peak = late.iter().fold(0.0, |a: Scalar, &b| a.max(b));
    assert!(rms(&late) < 0.3, "roll/pitch RMS {:.4} deg", rms(&late));
    assert!(peak < 0.5, "roll/pitch peak {peak:.4} deg");
    // Stability: 20 s later the error is the same size, not larger. A filter
    // that slowly walked off gravity would pass the first two assertions.
    assert!(
        rms(&late) < 1.5 * rms(&early),
        "wandered: {:.4} -> {:.4} deg",
        rms(&early),
        rms(&late)
    );
}

// ---------------------------------------------------------------------------
// Recovery from a wrong initial attitude, and the yaw invariant.
// ---------------------------------------------------------------------------

#[test]
fn a_thirty_degree_tilt_error_converges_while_yaw_is_left_alone() {
    // The phone spent half a minute in a pocket: the accelerometer never once
    // looked like gravity, so only the gyro ran, and an unknown turn-on bias
    // walked roll/pitch 30 degrees away from the truth. Then it comes out onto
    // a table. Structure of the run, all through the public API:
    //   1. initialise level,
    //   2. with the accelerometer gated out for the whole of steps 2 and 3,
    //      turn 40 degrees about the vertical so that yaw is at a value that is
    //      not the trivial zero,
    //   3. hold still for another 28 s while a 1 deg/s gyro bias tilts the
    //      estimate ~30 degrees off the truth,
    //   4. feed static level gravity and watch roll/pitch come back.
    // The assertion that matters is the last one: yaw must not have moved.
    //
    // Why the error is manufactured from an honest bias rather than by feeding
    // the gyro a rotation the device did not make. The recovery only means
    // something if the filter's *covariance* is honest about the error too.
    // One degree per second is exactly the turn-on bias sigma this crate
    // assumes (`BIAS_PRIOR_VARIANCE`), so after 30 unaided seconds the 30
    // degrees of tilt error sits at about one sigma of the filter's own
    // attitude covariance, and a Kalman update is entitled to act on it.
    // Fabricating a 30 deg/s gyro reading instead — the obvious way to write
    // this test — puts the truth a hundred sigma outside the covariance, and
    // the filter then does the only thing its model permits: it explains the
    // impossible rotation as a 9.6 deg/s gyro bias (ten sigma outside the
    // turn-on prior) and spends over thirty seconds unlearning it. Measured,
    // that route leaves 1.8 degrees of tilt after ten seconds of gravity. It
    // is the right answer to a question the model was never given the terms to
    // answer, and asserting on it would be measuring the rig.
    let config = OrientationConfig {
        static_threshold: 0.0, // no zero-rate update; this is about attitude
        ..OrientationConfig::default()
    };
    // One sigma of turn-on bias, about body x, which is what tilts the estimate.
    let bias = Vec3::x() * BIAS_SIGMA;
    let mut rig = Rig::clean(4).with_bias(bias);
    let mut filter = OrientationFilter::new(config);

    // Only this first sample gets to look like gravity; it sets the attitude.
    filter.integrate(&rig.sample(0.0, &So3::identity(), Vec3::zeros(), Vec3::zeros()));
    assert!(filter.is_initialized());

    // Blind from here on: 2 g on every accelerometer sample, far outside the
    // gate, so the estimate runs on the gyro alone.
    let blind = |filter: &mut OrientationFilter, rig: &mut Rig, t: Scalar, rate: Vec3| {
        let mut sample = rig.sample(t, &So3::identity(), rate, Vec3::zeros());
        sample.accel = Vec3::new(0.0, 0.0, 20.0);
        filter.integrate(&sample);
    };

    // The first sample already reports the turn rate. It has to: the truth is
    // turning at `yaw_rate` from t = 0 onwards, and a first sample reading zero
    // would describe a device that was still at t = 0 and turning by t = 0.01.
    // The filter integrates the trapezoid of the reported rates — deliberately,
    // see `propagate` — so it would return 39.9 degrees, correctly, for the
    // profile it was shown. Feeding a rate the truth does not have is a rig
    // bug, not a filter one.
    let yaw_rate = 40f64.to_radians() / 2.0;
    let mut t;
    for i in 0..=((2.0 * HZ) as usize) {
        t = i as Scalar * DT;
        blind(&mut filter, &mut rig, t, Vec3::z() * yaw_rate);
    }
    // 28 s of stillness, during which only the bias moves the estimate.
    for i in 1..=((28.0 * HZ) as usize) {
        t = 2.0 + i as Scalar * DT;
        blind(&mut filter, &mut rig, t, Vec3::zeros());
    }
    assert_eq!(filter.stats().gravity_accepted, 1, "init only");

    let yaw_before = filter.yaw();
    let level = So3::exp(&(Vec3::z() * yaw_before));
    let tilt_before = degrees(tilt_between(&filter.attitude(), &level));
    assert!(
        (tilt_before - 30.0).abs() < 1.0,
        "setup tilt {tilt_before}, wanted ~30 deg"
    );
    // And the filter says so: one sigma of its own attitude covariance is the
    // same 30 degrees, which is the whole point of building the error this way.
    let sigma_tilt = degrees(filter.covariance()[(0, 0)].sqrt());
    assert!(
        (0.5..2.0).contains(&(sigma_tilt / tilt_before)),
        "the covariance must own the error: {sigma_tilt:.1} deg sigma vs {tilt_before:.1} deg"
    );

    // Now present static level gravity for 10 s, noting the moment roll and
    // pitch are back so that the correction can be separated from what happens
    // afterwards.
    let settle_steps = (10.0 * HZ) as usize;
    let mut converged = None;
    for i in 1..=settle_steps {
        t = 30.0 + i as Scalar * DT;
        filter.integrate(&rig.sample(t, &level, Vec3::zeros(), Vec3::zeros()));
        if converged.is_none() && degrees(tilt_between(&filter.attitude(), &level)) < 0.05 {
            converged = Some((i, filter.yaw()));
        }
    }
    let (steps, yaw_at_convergence) = converged.expect("roll/pitch never converged");

    let tilt_after = degrees(tilt_between(&filter.attitude(), &level));
    assert!(
        tilt_after < 0.1 && steps < 40,
        "roll/pitch did not converge: {tilt_before:.2} -> {tilt_after:.4} deg in {steps} samples"
    );
    // The bias that caused it is observable from a static hold and is found.
    assert!(
        (filter.gyro_bias() - bias).norm() < 0.1 * bias.norm(),
        "bias {:?} not recovered from {bias:?}",
        filter.gyro_bias()
    );

    // The correction itself must not touch heading. It does leak a little, and
    // the bound is *relative* to the tilt removed rather than absolute, because
    // the leak is second order in that tilt.
    //
    // Why second order: the gravity update's unobservable direction is the
    // *estimated* vertical, not the true one. While the estimate is tilted by
    // theta those two differ by theta, so the null space is misaligned by
    // theta, and the component that escapes into world yaw goes as theta^2.
    // Measured over a 6.9-to-41.8 degree sweep, leak/tilt varies by a factor of
    // five while leak/tilt^2 stays inside a factor of two — see
    // `the_yaw_leak_is_second_order_in_the_tilt_removed`, which asserts the
    // scaling directly.
    //
    // An absolute threshold here would therefore be measuring how much tilt the
    // setup happened to build, not how well the filter protects heading.
    let yaw_moved = degrees(wrap_angle(yaw_at_convergence - yaw_before)).abs();
    assert!(
        yaw_moved < LEAK_COEFFICIENT * tilt_before * tilt_before,
        "gravity moved yaw by {yaw_moved:.7} deg while removing {tilt_before:.1} deg of \
         tilt, over the second-order budget of {:.6} deg",
        LEAK_COEFFICIENT * tilt_before * tilt_before
    );

    // Over the remaining ten seconds yaw does move, by about 0.07 degrees, and
    // it is worth being precise about why, because it is not gravity rotating
    // the estimate. While the estimate was 30 degrees over, body z was 30
    // degrees off the world vertical, so a slice of the *vertical* gyro bias
    // was genuinely visible to the accelerometer; the filter came away with
    // 1.3e-4 rad/s of it, under a hundredth of the real bias and under a
    // percent of the turn-on sigma, and a leftover bias integrates.
    //
    // So the sharp claim is that the drift *is* that bias: 0.073 deg observed
    // against 0.075 deg predicted, agreeing to under three percent. The couple
    // of thousandths left over is the same second-order leak measured above,
    // still running at a much lower level while the last of the tilt comes off
    // — so it is bounded by the same budget rather than by a number of its own.
    let drift = degrees(wrap_angle(filter.yaw() - yaw_at_convergence));
    let elapsed = (settle_steps - steps) as Scalar * DT;
    let from_bias = degrees(-filter.gyro_bias().z * elapsed);
    let unexplained = (drift - from_bias).abs();
    assert!(
        unexplained < 0.05 * from_bias.abs(),
        "yaw drifted {drift:.6} deg but the leftover bias only explains {from_bias:.6} \
         ({:.1}% unaccounted for)",
        100.0 * unexplained / from_bias.abs()
    );
    assert!(
        unexplained < LEAK_COEFFICIENT * tilt_before * tilt_before,
        "the {unexplained:.6} deg the gyro does not account for exceeds the \
         second-order leak budget of {:.6} deg",
        LEAK_COEFFICIENT * tilt_before * tilt_before
    );
    assert!(
        filter.gyro_bias().z.abs() < 0.01 * bias.norm(),
        "vertical bias picked up during the correction: {}",
        filter.gyro_bias().z
    );

    // And the yaw variance is at least what it was, because nothing observed it.
    assert!(filter.covariance()[(2, 2)] > 3.0);
}

/// The sharp form of "gravity does not rotate heading".
///
/// A bound on the yaw leak is only as good as the tilt it was measured at, so
/// this asserts the *scaling* instead. A correct ESKF leaks yaw only through
/// the misalignment between the estimated and true vertical, which is itself
/// first order in the tilt — so the leak is second order, and quartering the
/// tilt should sixteenth the leak.
///
/// A first-order bug — a measurement Jacobian that genuinely observes yaw, a
/// sign error, an update applied in the wrong frame — would show a *constant*
/// leak/tilt ratio and fail here, while comfortably passing any absolute
/// threshold chosen at a small tilt. That is the failure this test exists to
/// catch, and no single-point assertion can.
#[test]
fn the_yaw_leak_is_second_order_in_the_tilt_removed() {
    /// Build `blind_seconds` of unaided drift under a 1-sigma turn-on bias,
    /// then present level gravity and report `(tilt_removed, yaw_leaked)`.
    fn trial(blind_seconds: Scalar) -> (Scalar, Scalar) {
        let config = OrientationConfig {
            static_threshold: 0.0,
            ..OrientationConfig::default()
        };
        let bias = Vec3::x() * BIAS_SIGMA;
        let mut rig = Rig::clean(4).with_bias(bias);
        let mut filter = OrientationFilter::new(config);

        filter.integrate(&rig.sample(0.0, &So3::identity(), Vec3::zeros(), Vec3::zeros()));

        // Blind: 2 g on every sample keeps the accelerometer outside the gate.
        let blind = |filter: &mut OrientationFilter, rig: &mut Rig, t: Scalar, rate: Vec3| {
            let mut sample = rig.sample(t, &So3::identity(), rate, Vec3::zeros());
            sample.accel = Vec3::new(0.0, 0.0, 20.0);
            filter.integrate(&sample);
        };

        // Two seconds of yaw first. This is load-bearing, not scene-setting:
        // with the device left facing forward the body-x bias tilts it in a
        // plane containing the vertical, the geometry stays symmetric, and the
        // leak is identically zero — which would make the scaling
        // unmeasurable. Turning the device first breaks that symmetry, which
        // is also what makes the leak appear in the real scenario above.
        let yaw_rate = 40f64.to_radians() / 2.0;
        let mut t = 0.0;
        for i in 0..=((2.0 * HZ) as usize) {
            t = i as Scalar * DT;
            blind(&mut filter, &mut rig, t, Vec3::z() * yaw_rate);
        }
        for i in 1..=((blind_seconds * HZ) as usize) {
            t = 2.0 + i as Scalar * DT;
            blind(&mut filter, &mut rig, t, Vec3::zeros());
        }

        let yaw_before = filter.yaw();
        let level = So3::exp(&(Vec3::z() * yaw_before));
        let tilt = degrees(tilt_between(&filter.attitude(), &level));

        for i in 1..=((10.0 * HZ) as usize) {
            let now = t + i as Scalar * DT;
            filter.integrate(&rig.sample(now, &level, Vec3::zeros(), Vec3::zeros()));
            if degrees(tilt_between(&filter.attitude(), &level)) < 0.05 {
                let leaked = degrees(wrap_angle(filter.yaw() - yaw_before)).abs();
                return (tilt, leaked);
            }
        }
        panic!("roll/pitch never converged from {tilt:.1} deg");
    }

    let small = trial(10.0);
    let large = trial(40.0);
    let tilt_ratio = large.0 / small.0;
    assert!(
        tilt_ratio > 3.0,
        "the two trials must differ enough to separate the powers: {:.1} vs {:.1} deg",
        small.0,
        large.0
    );

    // A first-order leak would grow by `tilt_ratio`; a second-order one by its
    // square. Test against the geometric mean so the assertion is not a hair
    // either side of one of them.
    let leak_ratio = large.1 / small.1;
    let first_order = tilt_ratio;
    let second_order = tilt_ratio * tilt_ratio;
    let boundary = (first_order * second_order).sqrt();
    assert!(
        leak_ratio > boundary,
        "yaw leak grew {leak_ratio:.1}x for a {tilt_ratio:.1}x tilt increase \
         ({:.5} -> {:.5} deg). First order predicts {first_order:.1}x, second order \
         {second_order:.1}x — this looks first order, which means the gravity update \
         is genuinely observing yaw.",
        small.1,
        large.1
    );

    // And it stays negligible at both ends. The bound is quadratic for the same
    // reason the leak is: a constant leak/tilt ratio would itself be the wrong
    // shape, and would sit on a knife edge at one end of any wide sweep.
    // `LEAK_COEFFICIENT` has units of 1/deg and is ~4x the measured worst case.
    for (tilt, leak) in [small, large] {
        assert!(
            leak < LEAK_COEFFICIENT * tilt * tilt,
            "leak {leak:.6} deg exceeds the second-order budget \
             {:.6} deg for {tilt:.1} deg of tilt",
            LEAK_COEFFICIENT * tilt * tilt
        );
    }
}

// ---------------------------------------------------------------------------
// Gyro bias.
// ---------------------------------------------------------------------------

#[test]
fn a_constant_gyro_bias_is_recovered_and_the_error_stops_growing() {
    let config = OrientationConfig::default();
    let bias = Vec3::new(0.006, -0.004, 0.005);
    let mut rig = Rig::noisy(5, &config).with_bias(bias);
    let mut filter = OrientationFilter::new(config);
    let truth = So3::identity();

    let mut error_at = [0.0; 4];
    let steps = (30.0 * HZ) as usize;
    for i in 0..=steps {
        let t = i as Scalar * DT;
        filter.integrate(&rig.sample(t, &truth, Vec3::zeros(), Vec3::zeros()));
        for (k, mark) in [10.0, 20.0, 25.0, 30.0].iter().enumerate() {
            if (t - mark).abs() < DT * 0.5 {
                error_at[k] = degrees(angle_error(&filter.attitude(), &truth));
            }
        }
    }

    let relative = (filter.gyro_bias() - bias).norm() / bias.norm();
    assert!(
        relative < 0.10,
        "bias {:?} estimated as {:?} ({:.1}% off)",
        bias,
        filter.gyro_bias(),
        100.0 * relative
    );
    assert!(filter.stats().static_updates > 2000);

    // The error must plateau. Unestimated, this bias integrates to 0.5 deg/s of
    // yaw, so the last 5 s alone would add 2.5 deg.
    let growth = error_at[3] - error_at[2];
    assert!(
        growth.abs() < 0.1,
        "attitude error still growing: {error_at:?} deg"
    );
    assert!(error_at[3] < 0.5, "total error {:.3} deg", error_at[3]);
}

#[test]
fn a_turn_on_bias_larger_than_the_static_threshold_is_still_caught() {
    // The zero-rate update tests the *measured* rate, which a large turn-on bias
    // would otherwise push past the threshold forever — a filter that can never
    // observe the thing that is breaking it. The allowance widens by the current
    // bias uncertainty precisely so this bootstraps.
    let config = OrientationConfig::default();
    let bias = Vec3::new(0.030, 0.0, 0.0); // 1.7 deg/s, past the 0.02 rad/s gate
    let mut rig = Rig::noisy(6, &config).with_bias(bias);
    let mut filter = OrientationFilter::new(config);

    for i in 0..=((30.0 * HZ) as usize) {
        let t = i as Scalar * DT;
        filter.integrate(&rig.sample(t, &So3::identity(), Vec3::zeros(), Vec3::zeros()));
    }
    assert!(filter.stats().static_updates > 100);
    let relative = (filter.gyro_bias() - bias).norm() / bias.norm();
    assert!(relative < 0.15, "{:?} vs {bias:?}", filter.gyro_bias());
}

// ---------------------------------------------------------------------------
// The gate, and the ablation that shows it earns its place.
// ---------------------------------------------------------------------------

/// 6 m/s^2 at 3 Hz along world x is a hand shaking a phone, and it is the
/// disturbance the gravity gate exists for.
fn shake_at(t: Scalar) -> Vec3 {
    if t < 3.0 {
        Vec3::zeros()
    } else {
        Vec3::x() * (6.0 * (std::f64::consts::TAU * 3.0 * t).sin())
    }
}

/// Roll/pitch RMS, peak and gate rejections over the shaking half of a run.
///
/// 3 s of quiet to converge, then 5 s of shaking with the true attitude held
/// fixed, so every degree of error is the filter's.
fn shake_run(config: OrientationConfig, mut rig: Rig) -> (Scalar, Scalar, u64) {
    let mut filter = OrientationFilter::new(config);
    let truth = So3::identity();
    let mut errors = Vec::new();
    for i in 0..=((8.0 * HZ) as usize) {
        let t = i as Scalar * DT;
        filter.integrate(&rig.sample(t, &truth, Vec3::zeros(), shake_at(t)));
        if t >= 3.0 {
            errors.push(degrees(tilt_between(&filter.attitude(), &truth)));
        }
    }
    let peak = errors.iter().fold(0.0, |a: Scalar, &b| a.max(b));
    (rms(&errors), peak, filter.stats().gravity_rejected)
}

#[test]
fn gating_survives_a_shake_and_the_ungated_ablation_does_not() {
    // spec.md §6 requires ablations to be measurements, not assertions. Both
    // arms see the identical sample stream; only `gravity_gate` differs.
    //
    // The ablation is run twice, because the gate's benefit has two very
    // different sizes and reporting only one of them would misrepresent it.

    let gated_config = OrientationConfig::default();
    let open_config = OrientationConfig::ungated();
    // `Rig::noisy` reads only the two noise terms, which the ablation leaves
    // alone, so the two arms genuinely see the same stream.
    assert_eq!(gated_config.accel_noise, open_config.accel_noise);
    assert_eq!(gated_config.gyro_noise, open_config.gyro_noise);

    // Arm 1 — the gate against the disturbance alone, sensor noise switched off.
    //
    // This margin used to be 5.6x. It is now about 1.3x, and the shrinkage is
    // the point rather than a regression: `update_gravity` now inflates its
    // measurement noise by the sample's deviation from gravity, which is the
    // continuous form of the same judgement the hard gate makes discretely.
    // Once the filter distrusts an accelerating sample in proportion to how far
    // off it looks, a binary cutoff on top of that has much less left to do.
    //
    // The gate still earns its place — it bounds the worst case the soft
    // weighting only attenuates — so the assertion is that it helps, not that
    // it helps by the margin it needed to when it was the only defence.
    let (clean_gated, clean_gated_peak, rejected) = shake_run(gated_config, Rig::clean(7));
    let (clean_open, clean_open_peak, never_rejected) = shake_run(open_config, Rig::clean(7));
    assert_eq!(never_rejected, 0, "the ablation arm must reject nothing");
    assert!(rejected > 200, "the gate should have fired: {rejected}");
    assert!(
        clean_open > 1.15 * clean_gated,
        "the gate did not earn its place on the disturbance alone: \
         gated {clean_gated:.3} deg RMS vs ungated {clean_open:.3}"
    );
    assert!(clean_open_peak > 1.15 * clean_gated_peak);

    // Arm 2 — the same ablation on a device whose accelerometer has its own
    // 0.15 m/s^2 noise, paired over eight seeds because the outcome varies
    // several-fold between them and a single trial is not a measurement.
    //
    // The advantage halves, to ~2.7x, and the reason is a property of any
    // magnitude gate rather than of this one's threshold. A sample is admitted
    // while |a| is within `gravity_gate` of g, which at the 0.5 m/s^2 default
    // still allows a horizontal acceleration of sqrt(gate*(2g+gate)) = 3.2
    // m/s^2 — 17.9 degrees of apparent tilt — and admits it at *full weight*,
    // as trustworthy as a phone lying on a table. Sensor noise then moves
    // individual samples across that boundary, so the admitted set is a
    // different, randomly unbalanced subset of those 17.9-degree samples every
    // run, and the imbalance walks the estimate about half a degree. Four
    // times is not available here and no threshold makes it so; the fix would
    // be to weight each sample by its magnitude deviation, and that was
    // measured to remove the gate's advantage entirely rather than widen it.
    //
    // That fix has since been implemented — see `update_gravity` — and the
    // prediction held exactly. The assertion below now records the outcome.
    const TRIALS: u64 = 8;
    let mut gated_total = 0.0;
    let mut open_total = 0.0;
    let mut worst_gated = 0.0_f64;
    for seed in 7..(7 + TRIALS) {
        let (gated, _, fired) = shake_run(gated_config, Rig::noisy(seed, &gated_config));
        let (open, _, _) = shake_run(open_config, Rig::noisy(seed, &open_config));
        assert!(fired > 200, "seed {seed}: the gate should have fired");
        worst_gated = worst_gated.max(gated);
        gated_total += gated;
        open_total += open;
    }
    let gated_mean = gated_total / TRIALS as Scalar;
    let open_mean = open_total / TRIALS as Scalar;
    assert!(
        gated_mean < 1.0 && worst_gated < 1.5,
        "gated roll/pitch under a shake: mean {gated_mean:.3} deg RMS, worst {worst_gated:.3}"
    );
    // On a noisy device the gate is now a **wash**: 0.494 vs 0.480 deg RMS over
    // eight paired trials, the ungated arm marginally ahead and well inside the
    // spread between seeds.
    //
    // The comment above this block predicted precisely that, before the change
    // was made — "the fix would be to weight each sample by its magnitude
    // deviation, and that was measured to remove the gate's advantage entirely".
    // `update_gravity` now does that weighting, and the advantage duly went.
    // At a 5 m/s^2 deviation the inflated sigma is 29 degrees, so the sample is
    // effectively ignored whether or not a hard cutoff also rejects it.
    //
    // The gate is kept as a cheap backstop for absurd samples — free-fall, an
    // impact — where being certain beats being merely sceptical. But it is no
    // longer the primary defence, so this asserts only that it does no harm.
    // Asserting a benefit it no longer has would be asserting a fiction.
    assert!(
        gated_mean < 1.25 * open_mean,
        "the gate has become actively harmful, not merely redundant: gated \
         {gated_mean:.3} deg RMS vs ungated {open_mean:.3} deg RMS over {TRIALS} \
         paired trials"
    );
}

// ---------------------------------------------------------------------------
// Yaw drift, and L3 arresting it.
// ---------------------------------------------------------------------------

#[test]
fn yaw_drift_is_measurable_monotone_and_arrested_by_correct_yaw() {
    // spec.md §6 L1: "Yaw drift rate (deg/min), vision disabled." The zero-rate
    // update is switched off here because it would observe the bias directly and
    // there would be no drift left to measure — this is the vision-disabled,
    // hand-held case where the device is never quite still.
    let config = OrientationConfig {
        static_threshold: 0.0,
        ..OrientationConfig::default()
    };
    let drift_rate = 0.004; // rad/s about the vertical = 13.75 deg/min
    let bias = Vec3::new(0.0, 0.0, drift_rate);

    // One mark every 5 s. `i % marks` and not `(t / 5.0).fract() < DT/2`: at
    // t = 5 the fractional part of t/5 is under 0.005 for three consecutive
    // samples (t, t+0.01, t+0.02), so the fractional test records the same
    // instant three times and then compares those three against each other.
    let mark_every = (5.0 * HZ) as usize;
    let run = |seed: u64, corrections: bool| -> (Vec<Scalar>, OrientationFilter) {
        let mut rig = Rig::noisy(seed, &config).with_bias(bias);
        let mut filter = OrientationFilter::new(config);
        let mut marks = Vec::new();
        let mut next_correction = 1.0;
        for i in 0..=((60.0 * HZ) as usize) {
            let t = i as Scalar * DT;
            filter.integrate(&rig.sample(t, &So3::identity(), Vec3::zeros(), Vec3::zeros()));
            if corrections && t >= next_correction {
                // L3 reporting the true heading to one degree.
                filter.correct_yaw(0.0, 1f64.to_radians().powi(2));
                next_correction += 1.0;
            }
            if i > 0 && i % mark_every == 0 {
                marks.push(degrees(filter.yaw()));
            }
        }
        (marks, filter)
    };

    let expected = degrees(drift_rate) * 60.0;

    // The drift rate is a *measurement*, and one 60 s trial does not make one:
    // the gyro's own random walk puts 0.9 deg of noise on the 13.8 deg of
    // drift, and the filter's unobservable vertical-bias estimate wanders
    // another 1.7 deg on top, so a single trial has a ~14% standard deviation
    // (measured over 16 seeds: 10.9 to 17.4 deg/min). The original 5% bound on
    // one trial was never achievable — it is a third of one sigma. Eight
    // paired trials bring the standard error of the mean to ~5%, and the mean
    // is then checked against the truth at three of those, which is the
    // statement the noise model actually supports.
    const TRIALS: u64 = 8;
    let mut rates = Vec::new();
    let mut primary = None;
    for seed in 8..(8 + TRIALS) {
        let (free, filter) = run(seed, false);
        assert_eq!(free.len(), 12, "one mark per 5 s over 60 s");
        // Monotone on every trial: 5 s of drift is 1.15 deg against 0.26 deg
        // of gyro random walk over the same interval, a 4.4 sigma separation.
        for pair in free.windows(2) {
            assert!(
                pair[1] > pair[0],
                "seed {seed}: yaw drift must be monotone: {free:?}"
            );
        }
        rates.push(free[free.len() - 1] / 60.0 * 60.0);
        if primary.is_none() {
            primary = Some((free, filter));
        }
    }
    let n = rates.len() as Scalar;
    let mean = rates.iter().sum::<Scalar>() / n;
    let variance = rates
        .iter()
        .map(|r| (r - mean) * (r - mean))
        .sum::<Scalar>()
        / (n - 1.0);
    let standard_error = (variance / n).sqrt();
    assert!(
        standard_error < 0.06 * expected,
        "the drift-rate estimate is too noisy to be a measurement: \
         {standard_error:.3} deg/min on {expected:.3}"
    );
    assert!(
        (mean - expected).abs() < 3.0 * standard_error,
        "yaw drift {mean:.3} +/- {standard_error:.3} deg/min, expected {expected:.3}"
    );

    let (free, free_filter) = primary.expect("at least one trial");
    // Gravity cannot see the vertical bias, and the honest statement of that is
    // about the covariance, not the point estimate: the vertical bias variance
    // must still be the turn-on prior while the horizontal ones have collapsed,
    // and the estimate must stay a small fraction of its own sigma. (Asserting
    // the estimate itself is near zero would be wrong — an unobserved state is
    // free to wander inside its uncertainty, and this one wanders to ~0.15
    // sigma, which is 3x `drift_rate` and says nothing about the filter.)
    let bias_p = free_filter.bias_covariance();
    assert!(
        bias_p[(2, 2)] > 100.0 * bias_p[(0, 0)].max(bias_p[(1, 1)]),
        "the vertical bias should be the one thing gravity never learned: {bias_p}"
    );
    assert!(
        free_filter.gyro_bias().z.abs() < 0.3 * bias_p[(2, 2)].sqrt(),
        "vertical bias estimate ran away: {}",
        free_filter.gyro_bias().z
    );

    let (corrected, corrected_filter) = run(8, true);
    let free_final = free[free.len() - 1].abs();
    let corrected_final = corrected[corrected.len() - 1].abs();
    assert!(
        corrected_final < 0.05 * free_final,
        "correct_yaw did not arrest drift: {free_final:.3} -> {corrected_final:.3} deg"
    );
    // A heading reference makes the vertical bias observable through the
    // attitude/bias cross-covariance, which is the mechanism that stops the
    // drift rather than merely cancelling it once a second.
    assert!(
        corrected_filter.gyro_bias().z > 0.5 * drift_rate,
        "vertical bias not learned: {}",
        corrected_filter.gyro_bias().z
    );
    assert!(corrected_filter.stats().yaw_corrections >= 59);
}

// ---------------------------------------------------------------------------
// Covariance.
// ---------------------------------------------------------------------------

#[test]
fn the_covariance_is_a_valid_covariance_at_every_step() {
    // Everything at once: rotation, shaking, gating, gaps in the stream and yaw
    // corrections, with the 6x6 checked after every single sample.
    use wslam_core::covariance::is_valid_covariance;

    let config = OrientationConfig::default();
    let mut rig = Rig::noisy(9, &config).with_bias(Vec3::new(0.01, -0.008, 0.006));
    let mut filter = OrientationFilter::new(config);

    let mut truth = So3::identity();
    let mut t = 0.0;
    for i in 0..=((40.0 * HZ) as usize) {
        // A 3 s stall in the event stream at the 20 s mark.
        t += if i == (20.0 * HZ) as usize { 3.0 } else { DT };
        let rate = Vec3::new(
            0.6 * (0.7 * t).sin(),
            0.4 * (1.3 * t).cos(),
            0.9 * (0.31 * t).sin(),
        );
        truth = truth.plus(&(rate * DT));
        let linear = Vec3::new(4.0 * (5.0 * t).sin(), 0.0, 2.0 * (3.0 * t).cos());
        filter.integrate(&rig.sample(t, &truth, rate, linear));
        if i % 500 == 0 && filter.is_initialized() {
            filter.correct_yaw(wslam_orientation::gravity::yaw_of(&truth), 1e-3);
        }

        let p = filter.full_covariance();
        assert!(is_valid_covariance(&p, 1e-9), "step {i}: {p}");
        let smallest = p
            .symmetric_eigen()
            .eigenvalues
            .iter()
            .fold(Scalar::INFINITY, |a, &b| a.min(b));
        assert!(
            smallest > 0.0,
            "step {i}: not positive definite ({smallest})"
        );
    }
    assert_eq!(filter.stats().numerical_failures, 0);
    assert!(filter.stats().gravity_rejected > 0, "the shake should gate");
    assert!(
        filter.stats().gravity_accepted > 0,
        "and not gate everything"
    );
}

#[test]
fn the_yaw_variance_grows_monotonically_until_l3_speaks() {
    // The honest statement of what L1 knows: heading uncertainty only ever
    // increases. If this ever decreased without a correct_yaw call, L6 would be
    // publishing a confidence nobody earned.
    let config = OrientationConfig::default();
    let mut rig = Rig::noisy(10, &config);
    let mut filter = OrientationFilter::new(config);

    let mut previous = Scalar::NEG_INFINITY;
    for i in 0..=((20.0 * HZ) as usize) {
        let t = i as Scalar * DT;
        let truth = So3::exp(&(Vec3::new(0.2, 0.5, 0.84).normalize() * (0.4 * t)));
        filter.integrate(&rig.sample(
            t,
            &truth,
            Vec3::new(0.2, 0.5, 0.84).normalize() * 0.4,
            Vec3::zeros(),
        ));
        let v = filter.gravity_body();
        let yaw_variance = (v.transpose() * filter.covariance() * v)[(0, 0)];
        assert!(
            yaw_variance >= previous - 1e-12,
            "step {i}: yaw variance fell {previous} -> {yaw_variance}"
        );
        previous = yaw_variance;
    }

    filter.correct_yaw(0.0, 1e-4);
    let v = filter.gravity_body();
    let after = (v.transpose() * filter.covariance() * v)[(0, 0)];
    assert!(after < 0.01 * previous, "{previous} -> {after}");
}
