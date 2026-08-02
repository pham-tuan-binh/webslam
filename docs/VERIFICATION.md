# Verification

What each layer measures, against what, and where the number comes from.
Derived from spec.md §6, which is the section that matters.

The organising principle: **each layer has a different ground truth and must be
validated independently.** A system-level number tells you nothing about which
layer is broken.

---

## What is measured, and what is only unit-tested

Being blunt about this up front, because the project's whole thesis is that its
numbers can be trusted:

| Claim | Status |
|---|---|
| Lie-group algebra, covariance propagation, NEES/coverage machinery | **measured** against closed-form answers, 754 tests |
| Jacobians throughout | **measured** against central finite differences |
| L2 focal error + both required ablations | **measured** on a synthetic rotating rig with known focal |
| L1 angle error, gravity convergence, yaw drift | **measured** synthetically; needs the turntable for a device number |
| L3 ATE | **measured, and it fails.** 3.57 m RMSE on EuRoC MH_01_easy against ~0.05 m published. ~30% of frames produce no pose. See "First real-data result" below. |
| L4 relocalization + false-positive rate | **measured** synthetically; needs 7-Scenes for a public number |
| **L5 scale error %** | **synthetic only.** No metric truth source — see `rigs/README.md` |
| **L6 NEES / coverage on real data** | **synthetic only**, same reason |

The two rows in bold are the project's headline differentiators, and they are the
two that most need real ground truth. The machinery to compute them exists and is
tested; what is missing is a metric truth source to point it at.

---

## First real-data result: L3 does not meet spec

Run on EuRoC `MH_01_easy` (3682 frames, 184 s, published calibration, x86_64).

### Where it stands after twelve fixes

```
tier 1   15.9% no pose   ATE 3.185 m over the largest of 2 segments (95.6% coverage)
tier 2    9.9% no pose   ATE 2.862 m over the largest of 3 segments (55.3% coverage)
```

Published ATE for this sequence, ORB-SLAM3 Table II (median of 10 runs, Sim(3)
aligned, our comparison class):

| system | MH01 |
|---|---|
| ORB-SLAM3 mono | 0.016 |
| DSM | 0.039 |
| DSO | 0.046 |
| ORB-SLAM mono | 0.071 |
| SVO mono | 0.100 |

So the band is **0.016–0.100 m** and we are at **~3 m**: still one to two orders
of magnitude short. Vision-only buys no accuracy excuse here — the ORB-SLAM3
authors note that monocular's apparent edge over stereo is a 7-DoF-vs-6-DoF
alignment artefact, not a real one. What vision-only actually costs is
robustness.

**Frame loss is much improved** — 29.6% → 9.9% at tier 2 — and tier 2 now beats
tier 1, which is the ordering the architecture assumes.

### What the segmentation result ruled out

The trajectory now carries a coordinate-frame epoch and the harness evaluates
the largest segment. That was expected to be the dominant term. **It is not.**
The largest segment covers 95.6% of tier 1's poses and still scores 3.185 m, so
the error is genuine drift *within one continuous frame*, not an artefact of
splicing segments together.

This matters because it contradicts a plausible and well-argued hypothesis. The
splice was real, it was worth fixing, and fixing it did not move the number.

### Ranked remaining work

1. **Accuracy within a segment.** ~3 m of drift over 184 s against a 0.016–0.100 m
   band. Nothing in the harness is lying about it any more, so it is now
   directly attackable. The reference systems' answer is windowed local bundle
   adjustment over a covisibility graph plus loop closure with a Sim(3) pose
   graph; we have motion-only BA and no local BA at all.
2. **Relocalization still never fires.** Not the empty vocabulary, as first
   assumed — `add_keyframe` stores `landmarks: vec![None; n]`, so the
   verification PnP has no 3D points to work with. Populate keyframe landmarks
   from the tracker's local map first; the vocabulary is the second-order
   problem.
3. **Keyframe policy.** 470 keyframes over 3682 frames. ORB-SLAM2 gates on
   `mnMatchesInliers < 0.9 * nRefMatches` with a 20-frame floor at 20 fps; ours
   is a fixed 3-frame floor plus an absolute starvation trigger.

### The measurement was wrong in two ways, both now fixed

- **Camera frame vs body frame.** EuRoC's ground truth is `T_world_body`; we
  emit `T_world_camera`. The offset rotates with the body, so a single Sim(3)
  cannot absorb it — a hard **0.038 m floor** on every ATE we had reported,
  which is the same size as the whole published band.
- **Spliced segments.** One Sim(3) fitted across coordinate-frame
  discontinuities measures the seams. A research agent reproduced our original
  3.571 m headline from a *perfect* trajectory plus the splice alone.

---

## Tiers and cadence

| Tier | Needs | Runtime | Cadence | Where |
|---|---|---|---|---|
| **1 — Pure** | nothing | < 10 s | every commit | `cargo test --workspace` |
| **2 — Replay** | datasets | seconds–minutes | subset per commit, full nightly | `cargo xtask test two` |
| **3 — Rig** | turntable / strobe | minutes | nightly | manual; see `rigs/README.md` |
| **4 — Browser** | device matrix | slow, flaky | per milestone | manual + Playwright |

Tier 1 catches most bugs and must stay fast enough that nobody is tempted to
skip it. If it creeps past ~10 s, something belongs in Tier 2.

---

## Determinism is a prerequisite

A pipeline that reads wall-clock time and live frames cannot be regression-tested
at all. Three rules, enforced rather than requested:

| Rule | Enforced by |
|---|---|
| No `Date.now()` / `performance.now()` in the pipeline | `cargo xtask check-invariants`, which skips comments and test modules so it does not produce the false positives that get a check disabled; `HostClock` is a separate trait from `TimeBase` so "does this read the clock?" is one lookup |
| Every RNG is seeded and the seed logged | `DeterministicRng` has no entropy constructor and logs its seed on construction |
| The frame source is an interface | `FrameSource` trait, with `ReplayFrameSource` and the live shim |

CI additionally runs the whole suite twice and diffs the results. A single green
run hides an unseeded RNG; two identical runs do not.

---

## Per layer

### L0 — Clock

| | |
|---|---|
| **Ground truth** | Turntable + strobe. Cross-correlate gyro angular rate against image-derived rotation rate; the peak lag *is* the offset. The strobe gives independent absolute camera timing. |
| **Metrics** | Residual offset after correction (ms); **variance** of residual offset. |
| **Bar** | Beat the 30 ms native-access figure from arXiv:2001.00470. |
| **Reporting** | Per device. Aggregating hides the failure cases. |

Report the distribution, not a mean — the jitter is the thing we claim to fix.
If we cannot beat the bar, say so publicly and downgrade the `inertial`
ScaleSource. That is a stated outcome, not a failure mode.

### L1 — Orientation

| | |
|---|---|
| **Ground truth** | Turntable at a known rate; a level surface for gravity. |
| **Metrics** | Integrated angle error over 60 s (deg); roll/pitch error vs gravity (deg); yaw drift rate (deg/min) with vision disabled. |

### L2 — Intrinsics

| | |
|---|---|
| **Ground truth** | ChArUco calibration per device (`rigs/charuco.py`). Validation only — never read at runtime. |
| **Metric** | Relative focal error `|f̂ − f| / f`, distributed over ≥30 trials × ≥6 devices. |

**Required ablations.** These are gates, not nice-to-haves, because the
literature says both are failure modes:

1. With / without radial distortion in the model. Hayman & Murray (CVIU 2004):
   barrel distortion produces a sharply increasing *overestimate* of focal
   length, then outright failure.
2. With / without lever-arm modelling. Ji et al.: pure rotation is unachievable
   handheld — you rotate about your wrist, ~20 cm from the optical centre.
   `wslam_calib::SyntheticRig` models the pivot offset directly
   (`wrist_lever_arm_biases_focal_without_the_model`).
3. Across scene depth. Barrel distortion and lever-arm parallax interact with
   depth, so a single-depth result is not a result.
   (`lever_arm_bias_grows_as_the_scene_gets_closer` covers this synthetically.)

**Measured result, which differs from the citation.** The unmodelled arm
*under*estimates focal length by 5-8%, where Hayman & Murray report an
overestimate. They solve for the rotations jointly; we get them from L1, so the
error lands entirely on `f`. See docs/DECISIONS.md D11 before "correcting" the
test back.

### L3 — Tracking

| | |
|---|---|
| **Ground truth** | **EuRoC.** TUM-VI is fisheye (Kannala-Brandt) and unusable until a fisheye projection model exists — the loader refuses it rather than misreading its coefficients. |
| **Metric** | ATE after **Sim(3)** alignment — scale-free, because L3 does not claim scale. `wslam_core::math::umeyama`. |
| **Bar** | Match the native reference within noise. |

**Any divergence is a port bug, not an algorithm result.** Keeping that claim
checkable is why the GPU kernels each have a CPU reference and an equivalence
test, and why browser runs are fed a *recorded* frame source — so a failure
identifies whether it was the numerics or the plumbing.

### L4 — Map

**Relocalization.** Build a map, move away, occlude, return via a *different*
path, measure recovery. Success rate as a function of viewpoint
change (baseline distance and view angle); time-to-relocalize; pose error
immediately post-recovery. Plus 7-Scenes for a comparable public number.

**Loop closure.** ATE on EuRoC with and without closure — the standard
comparison, which also validates the port.

**False-positive rate, measured separately and reported prominently.** A false
positive corrupts the map irrecoverably and is *worse than no loop closure at
all*. Sample non-revisited sequences and count spurious matches surviving
geometric verification. **Target zero**; anything above zero means raising the
verification threshold, not shipping.

**Resource envelope.** Map memory growth (MB/min) — a phone tab is killed if
this is unbounded. Backend latency distribution, and confirmation the frontend
never stalls on it: **p99 frame time, not mean**. Under the single-threaded
default (docs/DECISIONS.md D2) this is the design's primary risk, so the budget
is enforced in code rather than by convention.

### L5 — Scale (the headline metric)

| | |
|---|---|
| **Ground truth** | **None currently.** The robot-arm rig was removed as out of scope (`rigs/README.md`), and it was the only source of exact metric 6-DoF. Validated against synthetic trajectories with known scale. |
| **Primary metric** | Scale error % as a function of time-since-init. **Reproduce the Campos curve**: report 2 s and 10 s numbers against their 5% / 1%. |
| **Design** | Paired — every ScaleSource evaluated on *identical* trajectories, so comparisons are within-trajectory rather than across runs. |

**Report the excitation dependence explicitly.** A single aggregate number for
inertial scale is misleading, because the theory says accuracy collapses as
translational acceleration vanishes. `wslam_scale`'s tests sweep synthetic
excitation and assert the error falls as excitation rises, and that the source
returns `None` under a static hold. Measuring the *real* curve needs a metric
truth source — see the gap noted under L5.

### L6 — Uncertainty calibration

We claim the covariance is meaningful. Prove it.

| Metric | Definition | Pass |
|---|---|---|
| **NEES** | Normalised estimation error squared over ≥100 trials | Mean inside the chi-squared interval for `n·k` degrees of freedom |
| **Coverage** | Do 95% intervals contain truth 95% of the time? | Within 2% of nominal at 68/95/99 |

`wslam_core::covariance::ConsistencyAccumulator` computes both, and
`ConsistencyReport` distinguishes **overconfident** from merely conservative —
consistently above the bound is worse than no covariance at all, while below it
is only unhelpful.

The Tier-1 suite includes a self-test: a synthetically *perfectly calibrated*
estimator must pass, and a deliberately overconfident one must be flagged. If
those two ever disagree with expectation, the harness is wrong rather than the
estimator, and that distinction is worth being able to make.

---

## System level

- **Device × OS matrix.** iOS 26+ (the WebGPU floor) and Android Chrome.
  **Report per cell, never pooled** — device-specific quirks were 8th Wall's
  actual moat and will be our actual bug surface. `rigs/capture.py` records the
  cell alongside every measurement so pooling is not the path of least
  resistance.
- **Thermal.** 15-minute sustained runs logging pose error and frame rate against
  elapsed time. Report time-to-degradation per device.
- **Rolling shutter.** Sweep arm velocity, plot error against angular rate.
  Establishes and documents the motion envelope rather than pretending it does
  not exist.

## Statistical design

- Fix the primary metric — **scale error at 10 s** — and pre-register it before
  running anything.
- Power-analyse trial counts for the effect sizes that matter. Distinguishing 1%
  from 2% scale error needs a specific N; guessing wastes weeks.
- Paired trials wherever conditions can share a trajectory.
- Correct for multiple comparisons across the ScaleSource × device grid.

## Logging to rerun

Nightly runs record a rerun session as a build artifact. When a regression
fires, scrub the failing run visually instead of bisecting from a single ATE
number. `harness/viewer` owns the logging helpers; CI uploads the `.rrd`.


## Accuracy on EuRoC MH_01 — measured, 2026-08-02

All figures are **segment** ATE (Sim(3)-aligned within each coordinate-frame
epoch), tier 2, full 3682-frame sequence. Whole-trajectory ATE is reported by
the harness but measures the seams between epochs, not tracking quality.

| change | keyframes | segment ATE | frames with no pose |
|---|---|---|---|
| baseline (absolute starvation trigger, local BA on) | 1002 | 3.20 m | 11.2% |
| relative keyframe policy (ORB-SLAM2 `thRefRatio`), BA off | 281 | 2.12 m | 15.0% |
| + undistorted pixels into every geometry consumer | 275 | **0.32 m** | 26.2% |
| + local BA, 10-keyframe window | 308 | 2.15 m | 17.3% |

Published monocular band on this sequence: 0.016–0.100 m. We are at 0.32 m —
roughly 3–20x off, against ~200x before this work.

### The observation-model bug dominated everything

Storing and consuming **distorted** pixels while `pnp`, `triangulate`, and
`motion_ba` all document undistorted input was worth a 6.6x error reduction on
its own. EuRoC's `k1 = -0.283` displaces the median pixel by ~22 px, so 85.7% of
the image sat beyond the 3 px RANSAC threshold; PnP was systematically
discarding the peripheral observations with the most geometric leverage.

The bootstrap was worse than consistently wrong — it paired a distorted stored
observation with an undistorted current one, so the two views disagreed about
the camera model. A partial earlier fix had introduced `Feature::px_undist` and
wired it at two sites, which is what made the remaining ones hard to see.

**The lesson generalises.** Every measurement taken while that bug was live is
uninterpretable, including the local-BA ablations that motivated the keyframe
policy work. Bundle adjustment minimises reprojection error *under the assumed
projection*; with a 22 px model error the cost minimum is not the true geometry,
and the solver converges confidently to the wrong answer while reporting a
falling cost. No internal consistency check can catch this — only alignment
against external truth.

### Local BA is off by default, and why

It made accuracy worse in all four configurations tried. The solver is
unit-tested and recovers perturbed landmarks on synthetic scenes (>10x error
reduction), so the defect is in window assembly or write-back, not in the
optimisation. Two real bugs were found and fixed along the way and are worth
keeping regardless:

- **Scale gauge.** A monocular reconstruction is gauge-free in *seven*
  directions, not six. Two fixed poses pin scale only in proportion to their
  separation relative to scene depth. With near-coincident anchors the solver
  inflated the trajectory ~3000x (Sim(3) scale 0.0003) while reducing
  reprojection cost — a uniform expansion of poses *and* points reprojects to
  nearly identical pixels, so cost is blind to it.
- **Per-solve bounds cannot fix a systematic bias.** Capping each solve at 1.5x
  still left the trajectory 8.7x too large: a 1.01x bias compounds over ~300
  windows. Holding landmarks fixed when scale is unobservable removes the
  freedom instead of bounding its abuse.

### Known-unexplained

The Sim(3) alignment scale sits at 0.30 rather than near 1.0, i.e. the
reconstruction is ~3.4x larger than truth. For a monocular system scale is
arbitrary and Sim(3) alignment absorbs it, so this is not necessarily a defect —
but it has not been explained, and it is not being claimed as correct.


## GPU front-end and L4, 2026-08-02 (second pass)

Production configuration measured on EuRoC MH_01, tier 2, all 3682 frames, on a
desktop NVIDIA GPU:

```
--gpu --max-features 1000 --vocab <trained>
```

| | BA off | BA window 10 |
|---|---|---|
| frames with no pose | **0.1%** | **0.1%** |
| coordinate frames | **1 (100% coverage)** | **1 (100% coverage)** |
| ATE (whole trajectory) | **0.291 m** | 0.598 m |
| RPE inter-frame (50 ms) | 0.0147 m, 0.130 deg | **0.0057 m, 0.053 deg** |
| RPE 1 s | 0.0556 m, 0.316 deg | 0.0546 m, 0.196 deg |
| frame time median / p99 | **1.92 / 18.3 ms** | 2.01 / 33.3 ms |
| loops accepted / rejected | 30 / 846 | 52 / 782 |

Local BA buys a **2.6x better inter-frame RPE** and costs **2x the absolute
error**. Which one to ship depends on whether the consumer feels jitter or
drift; neither default is obviously right, so BA stays off and the trade is
documented rather than decided silently.

### The GPU front-end was declared but never dispatched

`wslam-gpu` implemented `upload`, `build_pyramid`, `detect_corners`,
`track_flow` and `swap`, with tests, and `Backend::Gpu` was a label with no call
behind it. Wiring it up moved the median frame from 50.5 ms to 1.9 ms — a 26x
reduction — because the four hot stages are exactly the ones that scale with
feature count. Three bugs surfaced while connecting it, all at the boundary:

1. **Seed semantics.** The CPU `klt::track(prev, next, points, guesses)` takes
   source positions *and* a separate initial guess. `track_flow(points)` takes
   only source positions. Passing `guesses` did not seed the search, it lied
   about where each feature came from. Fixing it took frame loss from 52.6% to
   **0.1%** and made the trajectory continuous for the first time — one
   coordinate frame instead of ten.
2. **No backward pass.** Reporting `fb_error: 0.0` made the forward-backward
   gate accept everything; ATE was 2.36 m against 0.085 m for the CPU. The pass
   is now run for real by swapping the pipeline's frame sets.
3. **Tolerance transfer.** The f32 kernel's round trip is noisier than the f64
   reference, so the CPU's 1 px tolerance rejected good tracks. 3x recovers the
   CPU rejection rate.

### L4 was inert for a reason unrelated to L4

`MapDb::new(Arc::new(Vocabulary::empty()))`. `Vocabulary::transform` early
returns an empty bag when it has no words, so every keyframe's descriptor
signature was empty and place recognition could never propose a candidate — 0
proposed over 3682 frames with real revisits. The comment at that line asserted
the opposite ("still supports relocalization within a session"), which is why it
survived. With a trained vocabulary installed, 30 loops are accepted.

Keyframe landmark association was also missing (`landmarks: vec![None; n]`), so
even a proposed loop had no 3D correspondences to verify against.

**Caveat on the vocabulary:** it is trained on MH_01, the same sequence it is
evaluated on, because no second sequence is available on the test machine. That
inflates recall relative to a held-out vocabulary and the loop counts above
should be read as an upper bound.

### Loop closure corrections still do not reach the trajectory

Accepting 30 loops changed the reported ATE by zero digits (0.2906 both ways):
poses are emitted write-once, so a correction to a *keyframe* cannot propagate
to frames already emitted. `refined_trajectory` implements ORB-SLAM2's
`mlRelativeFramePoses` to fix this, and is **off by default because it is
defective**. With loop closure disabled it must reduce to the identity — instead
absolute error holds (0.2914 vs 0.2906) while inter-frame RPE degrades from
0.0147 m / 0.130 deg to 0.0313 m / 0.834 deg. Something moves a frame's anchor
between recording and readout. Until that is found it adds noise instead of
removing drift.

### Against the goal

- **<30 ms on a phone.** 1.92 ms median / 18.3 ms p99 on a desktop GPU. Not yet
  demonstrated on a phone, and a mobile GPU is several times slower; this is
  plausible but **unverified**, and no phone measurement exists.
- **0.05 m absolute.** Not met. Best is 0.291 m. The remaining error is
  uncorrected drift; the mechanism to remove it (loop closure) now fires but
  cannot reach the trajectory, per the defect above.
- **Relative pose.** 0.0057 m / 0.053 deg between consecutive frames with local
  BA on, 0.0147 m / 0.130 deg without.


## Accuracy pass three, 2026-08-02 — fixed context keyframes

Shipping defaults, EuRoC MH_01, tier 2, `--gpu --max-features 1000`:

```
frames      3682 over 184.0s, 0.1% produced no pose, 1 coordinate frame (100%)
ATE         0.3764 m
RPE         0.0048 m / 0.056 deg inter-frame, 0.0358 m / 0.171 deg over 1 s
frame time  median 1.96 ms, p99 23.70 ms
```

### What moved, and what did not

**Fixed context keyframes were the missing piece of local BA.** A landmark
observed inside the window is also observed by older keyframes outside it. With
those keyframes absent from the problem, nothing held the landmark to the map it
already belonged to, so each solve could rescale the local reconstruction freely
— local consistency bought with global scale drift. Including them as fixed
poses is what ORB-SLAM2's `LocalBundleAdjustment` does, and it recovers most of
the loss:

| context keyframes | ATE | RPE inter-frame | RPE 1 s |
|---|---|---|---|
| 0 | 0.598 m | 0.0057 m | 0.0546 m |
| 10 | 0.488 m | 0.0046 m | 0.0493 m |
| 25 | 0.393 m | 0.0047 m | 0.0372 m |
| 50 | 0.376 m | 0.0050 m | 0.0370 m |
| 100 | 0.380 m | 0.0049 m | 0.0365 m |

Saturates around 25-50.

**The frame-time tail is set by BA, not by tracking.** Median is ~2 ms in every
configuration; the p99 is whichever keyframe ran the largest solve. Window 40
reaches ATE 0.307 m but a 325 ms p99. Capping LM iterations at 4 with window 10
holds ATE at 0.376 m for a **23.7 ms p99**, which is what ships.

**Local BA still costs absolute accuracy.** 0.376 m with BA against 0.291 m
without. It ships on because inter-frame RPE is 3x better (0.0048 m vs
0.0147 m) and 1 s RPE 1.5x better; two of three metrics improve substantially.
The regression is real and is recorded here rather than hidden.

### Falsified along the way

- **Motion-only BA** (landmarks held fixed) was proposed as a way to remove
  scale drift by removing the gauge freedom entirely. It is much worse: ATE
  2.34 m, inter-frame RPE 1.706 deg. With the map frozen, poses are dragged onto
  structure triangulated from short baselines. Landmark refinement is not
  optional.
- **More features.** Beyond ~1000-1500 accuracy degrades and then flatlines
  (2500 and 3500 give byte-identical results) because the corner grid caps
  supply at one winner per cell.
- **A tighter per-solve scale guard.** 1.5x, 1.02x and 1.001x all leave ATE
  within noise of each other. Bounding a systematic bias does not remove it.

### The `refined_trajectory` defect was a frame convention, and is fixed

The emitted path converts `T_world_camera` to `T_world_body` through the
camera-IMU extrinsic; the refined path did not. A single Sim(3) cannot absorb a
lever arm that rotates with the body, which is exactly why absolute error
survived while inter-frame RPE collapsed from 0.130 deg to 0.834 deg. With the
conversion applied, the invariant holds exactly: with zero loop closures the
refined trajectory reproduces the emitted one to every reported digit.

That was the third appearance of the same camera-vs-body confusion in this
codebase, in a third code path.

### Loop closure is correct and still cannot be used

30 loops are accepted and they make things worse — ATE 0.291 m to 0.851 m. The
cause is structural, not a tuning problem: **the pose graph is SE(3)-only.** In
a monocular map scale drifts, so the two ends of a loop are at different scales,
and an SE(3) edge asserts that they are not. That contradiction propagates
through the whole graph. This is precisely why ORB-SLAM uses Sim(3) for
monocular loop closing. `wslam-core` already has the `Sim3` type; the pose graph
and the loop measurement do not use it. That is the next substantial piece of
work and the main remaining path to a sub-0.1 m absolute error.


## The WebGPU front-end cannot run in a browser yet, 2026-08-02

Wiring `wslam-gpu` into the wasm build hard-locks the tab on the first frame.
Confirmed by remote-debugging the deployed page: after clicking Start,
`Runtime.evaluate('1+1')` stops returning and never recovers. Isolated by
building the identical page with and without the `gpu` feature — with it, the
main thread is dead; without it, the page stays responsive, `start()` resolves,
and the camera streams.

The cause is `read_back` in `crates/wslam-gpu/src/lib.rs`:

```rust
slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r.is_ok()); });
device.poll(wgpu::PollType::wait_indefinitely())?;
match rx.recv() { ... }
```

Correct natively. On wasm it is a guaranteed deadlock: WebGPU only completes
work through the browser event loop, and blocking the single JS thread on
`rx.recv()` stops that loop, so the map callback can never fire. Both
`detect_corners` and `track_flow` go through it, so the deadlock is immediate.

`GpuContext::new_blocking` is already `#[cfg(not(target_arch = "wasm32"))]`
with a comment saying blocking deadlocks the event loop. The same reasoning
applies to readback and was not carried through.

Lifting this needs `read_back` to become async and that asynchrony to reach
`Tracker::process`, which is a signature change through L3. Until then a
`compile_error!` in `wslam-wasm` rejects `gpu` + `wasm32` outright: the failure
mode is an unresponsive tab, not a slow one, so it must not be reachable by
accident.

**Consequence for the published demo.** It runs the CPU reference front-end.
None of the GPU figures — 1.96 ms median, ATE 0.376 m — describe it. The CPU
path measured ~50 ms per frame on a desktop at 1000 features, and the wasm
build defaults to 250.
