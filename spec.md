# web-slam

**A 6-DoF pose source for stock mobile browsers, with an explicit metric anchor and calibrated uncertainty.**

Status: proposal
Owner: TBD
Last updated: 2026-08-01

---

## 1. What we are building

A JavaScript/WASM library that produces 6-DoF camera pose at 30–60 Hz inside an unmodified mobile browser — including iOS Safari — from `getUserMedia` and `DeviceMotion` alone.

Two properties distinguish it from every existing option:

1. **Metric scale comes from a declared, pluggable source.** The library never silently guesses scale. Callers choose an anchor and accept its tradeoffs.
2. **Every pose is emitted with a covariance**, and that covariance is validated to be statistically calibrated rather than decorative.

### Non-goals

Explicitly out of scope, and we should resist all of them:

- Dense mapping or reconstruction — our map is sparse keyframes and landmarks, nothing renderable
- Plane detection, image targets, face tracking, occlusion, rendering
- Multi-user / cloud-shared maps
- Matching ARKit/ARCore accuracy
- Working on every consumer phone in existence

That last one is what consumed 8th Wall for seven years. We are building a component, not a platform.

---

## 2. Why this exists

**WebXR is unavailable on iOS.** Safari implements no `immersive-ar`, and because every iOS browser is WebKit underneath, no browser install fixes it. Apple has announced no timeline. This is the single reason browser-based world tracking is still an open problem, and it is not going to resolve itself.

**Monocular metric scale is unobservable.** This is a theorem, not a gap in the literature. A scene twice as large at twice the distance produces pixel-identical images. Formal observability analysis of monocular visual-inertial systems establishes four unobservable directions under generic motion, and shows metric scale becomes unobservable whenever translational acceleration vanishes — which is exactly how people hold phones.

Therefore scale always comes from a *ruler*, and there are only four:

| Ruler | Accuracy | Cost |
|---|---|---|
| Inertial (double-integrated acceleration) | ~1% given excitation | needs motion, bias estimation, tight time sync |
| Known object in scene | exact | object must be visible |
| Known baseline (stereo) | exact | **unavailable — browser gives one stream, no extrinsics** |
| Learned monocular prior | several %, domain-correlated | model download, GPU |
| User declares one distance | exact | one tap |
| **Persisted metric map** | inherits its anchor | must relocalize; requires L4 |

The last row is why mapping is in scope. Anchor scale *once* by any of the above, persist the map, and every subsequent session recovers metric by relocalizing. This converts scale from a hard per-session estimation problem into a one-time one.

Every metric system that has ever shipped uses one or more of these. Our architecture makes the choice explicit instead of burying it.

**Four of our five layers are ports of settled work.** One is not. Scoping to that reality is the point of this document.

---

## 3. Public API

**This is the product. Everything below it is implementation.**

Design rules, in priority order:

1. **The default path is one screen of code.** A developer who wants pose and nothing else should never read about ScaleSources or sensor tiers.
2. **Progressive disclosure.** Every advanced capability is reachable, none is mandatory.
3. **No silent assumptions.** If scale is guessed, the API says which ruler guessed it and how much it trusts itself.
4. **The debug surface is first-class**, because our own viewer and demo are its first consumers.

### The 90% case

```ts
import { WebSlam } from 'web-slam';

const slam = await WebSlam.create({ video: videoEl });

// iOS requires a user gesture before motion sensors — this is a hard
// platform constraint, so start() is explicit and must be called from
// a click handler. There is no way to hide this.
startButton.onclick = () => slam.start();

renderLoop(() => {
  const pose = slam.currentPose();   // pull at your frame rate
  if (pose) camera.matrix.fromArray(pose.matrix);
});
```

Defaults: sensor tier 2 (vision + loose orientation), scale source `none` (up-to-scale), map enabled with relocalization.

### Pull vs push

Both, and the distinction is documented, not incidental:

- **`slam.currentPose()`** — pull. For renderers. You want the freshest pose at *your* draw time, not every camera frame.
- **`slam.onPose(cb)`** — push, fires per tracked frame. For recorders, teleop transmitters, anything that must not drop samples.

### Pose

```ts
interface Pose {
  timestamp: number;            // mapped into performance.now() domain
  position: Vec3;               // metres when scale.source !== 'none'
  rotation: Quaternion;
  matrix: Float32Array;         // 4x4 column-major, renderer-ready
  covariance: Float64Array;     // 6x6, [translation, rotation]
  scale: { source: ScaleKind; variance: number };
  state: TrackingState;
}

type TrackingState =
  | 'initializing'
  | 'tracking'
  | { limited: 'excessive-motion' | 'insufficient-features' | 'low-light' }
  | 'relocalizing'
  | 'lost';
```

Covariance and scale provenance travel *with* every pose. They are not queried separately, because separate queries get skipped.

### Scale, declared explicitly

```ts
await WebSlam.create({
  video: videoEl,
  scale: ScaleSource.fiducial({ family: 'apriltag36h11', sizeMeters: 0.1 }),
});
```

```ts
ScaleSource.none()                        // up-to-scale, honest default
ScaleSource.declared()                    // one user tap on a known distance
ScaleSource.fiducial({ family, sizeMeters })
ScaleSource.map(savedMap)                 // inherits the anchor's variance
ScaleSource.learned({ model })            // opt-in, downloads weights
ScaleSource.inertial()                    // requires tier 3; throws if L0 unavailable
```

### Events

```ts
slam.onState((state, prev) => ...);
slam.onRelocalize(({ atTimestamp, mapPoseId }) => ...);
slam.onLoopClosure(({ accepted, candidateId }) => ...);
```

### Map persistence

```ts
const bytes = await slam.saveMap();                 // Uint8Array, caller stores it
const slam2 = await WebSlam.create({
  video: videoEl,
  scale: ScaleSource.map(bytes),
});
```

### Debug surface

Namespaced, tree-shakeable, and **explicitly unstable** — versioned separately from the core API so the viewer can move fast without pinning us.

```ts
slam.debug.landmarks();      // Float32Array xyz
slam.debug.keyframes();      // poses + ids
slam.debug.trajectory();
slam.debug.features();       // 2D, with per-feature state for overlay colouring
slam.debug.poseGraph();      // edges, including rejected loop candidates
slam.debug.timings();        // per-stage: upload/pyramid/corners/flow/pnp
```

### Integration helper

A `web-slam/three` entry point providing a camera-sync helper. Adoption friction is mostly in the first fifteen minutes.

### Freeze it at M0, against a mock

**The public demo is written at M0 against a stub implementation that returns synthetic poses.** No tracking, no map, canned data. The point is to validate the API with a real consumer before any of it exists, and to make the interface expensive to change afterwards rather than cheap to drift.

If the demo is awkward to write against the mock, the API is wrong, and M0 is when that costs nothing.

---

## 4. Architecture

Six layers. Layers 0–3 are scene-agnostic and reusable. Layer 4 is where domain assumptions enter.

```
L6  Output          pose + covariance + scale provenance
L5  ScaleSource     ← pluggable, the only opinionated layer
L4  Map             keyframes, place recognition, pose graph  ← async backend
L3  Tracking        sparse KLT + PnP frontend, pose up to scale
L2  Intrinsics      focal length, estimated once at init
L1  Orientation     gyro + gravity, drift-free in roll/pitch
L0  Clock           unified timebase for camera and IMU
```

L3 and L4 run on separate threads. The frontend must never block on the backend.

### Sensor tiers

Sensor use is declared configuration, not an assumption — the same discipline we apply to scale.

| Tier | Sensors | Needs L0? | Status |
|---|---|---|---|
| 1 | Vision only | no | fallback when motion permission is denied |
| **2** | **Vision + loose orientation** | **no** | **baseline — what we ship** |
| 3 | Tight visual-inertial | yes | optional; research track R1 |

Tier 2 gets gravity direction, an orientation prior for tracking prediction, and survival through brief vision failure — none of which require sub-frame temporal alignment. The *only* thing tight coupling adds is inertial metric scale, and scale is already pluggable. This is why L0 is off the critical path.

### L0 — Clock

The foundation. Everything above it is wrong if this is wrong.

Approach: fit a linear clock model over `DeviceMotion` event *index* rather than trusting per-event timestamps (events are generated on a regular hardware cadence and only *delivered* with event-loop jitter). Do the same on the video side using `requestVideoFrameCallback`'s `mediaTime`, which rides the media clock rather than wall clock. Then estimate the residual constant offset online as a filter state.

### L1 — Orientation

Gyro integration with accelerometer gravity correction. Drift-free in roll and pitch; yaw drifts slowly and is arrested by L3. Three of six DoF, solved, no vision required.

### L2 — Intrinsics

Focal length from rotation-compensated homographies during a short init pan, using gyro-known rotation. See §4 for the two failure modes that make this harder than it looks.

### L3 — Tracking

Sparse feature tracking (KLT), PnP against the active local map. WebGPU compute shaders for pyramid/corner/flow; read back a few hundred points, never a full image. Pose up to scale. Runs every frame, on the critical path.

### L4 — Map

Three capabilities, staged by value-to-cost:

**(a) Keyframe map + relocalization — mandatory.** Tracking loss is not an edge case on a phone; it is a routine event from occlusion, pocketing, or fast motion. Without relocalization the session and the user's anchor are destroyed. Sparse keyframes with binary descriptors plus a bag-of-words database. Cheap, well-trodden, and the single largest UX win in the project.

**(b) Loop closure + pose graph — second.** Bounds global drift, particularly yaw, which L1 and L3 only arrest locally. Requires a graph solver in WASM.

**(c) Map persistence — third.** Serialise the anchored map so later sessions relocalize into metric. This is what makes the map a ScaleSource.

**Deployment constraint:** a threaded backend needs pthreads → `SharedArrayBuffer` → COOP/COEP headers. That breaks third-party embedding. Either accept single-threaded backend optimisation at reduced rate, or accept the header requirement and document it loudly. **Decide before M4.**

### L5 — ScaleSource

A single interface with four implementations:

```ts
interface ScaleSource {
  readonly kind: 'inertial' | 'fiducial' | 'learned' | 'declared' | 'map';
  estimate(window: StateWindow): { scale: number; variance: number } | null;
}
```

Consumers pick. A teleop client may accept `declared`; a data-collection rig requires `fiducial`. `map` inherits the variance of whatever anchored it, plus relocalization error — it must not report itself as more certain than its origin.

### L6 — Output

```ts
interface Pose {
  R: Quaternion;  t: Vec3;
  covariance: Float64Array;  // 6x6
  scaleSource: ScaleSource['kind'];
  scaleVariance: number;
  initAgeMs: number;
}
```

Emitting a naked transform forces every consumer to guess how much to trust it. We won't.

---

## 5. Prior art

### Time synchronisation — theory settled, browser instantiation unclaimed

- Li & Mourikis, *Online Temporal Calibration for Camera-IMU Systems: Theory and Algorithms*, IJRR 2014. Puts the offset `td` directly in the EKF state, proves recoverability, identifies degenerate motions where estimation should be suspended.
- Qin & Shen, *Online Temporal Calibration for Monocular Visual-Inertial Systems*, arXiv:1808.00692 (IROS 2018). Shipped in VINS-Mono.
- Furgale et al., **Kalibr** — offline, continuous-time spline batch estimator, requires a target.
- Huai et al., *The Mobile AR Sensor Logger for Android and iOS Devices*, arXiv:2001.00470. **Measured up to 30 ms camera-IMU offset on real phones with full native API access.** Our calibration bar. The browser will be worse.

**Gap:** all of the above estimate a *constant* offset. Our problem is per-sample delivery jitter on top of an unknown constant. No published treatment for the browser environment.

### Intrinsics from rotation — solved 1994–2001, with real caveats

- Hartley, *Self-calibration from multiple views with a rotating camera*, ECCV 1994.
- de Agapito, Hayman & Reid, *Self-Calibration of Rotating and Zooming Cameras*, IJCV 45(2):107–127, 2001. Infinite homography constraint; linear method under zero-skew or square-pixel assumptions.
- Hayman & Murray, *The impact of radial distortion on the self-calibration of rotating cameras*, CVIU 2004. **Barrel distortion produces a sharply increasing overestimate of focal length, then outright failure.** Phone wide cameras are barrel-distorted.
- Ji et al., *Self-Calibration of a Rotating Camera With a Translational Offset*. **Pure rotation is unachievable handheld** — you rotate about your wrist, ~20 cm from the optical center, which injects translation.

**Mitigation:** the lever arm is precisely the camera-IMU extrinsic translation that VI systems already estimate. Fold it in rather than treating it as a separate hack. Both caveats must be explicit ablations (§5).

### Scale via inertial — settled, and better than we assumed

- Martinelli; Dong-Si & Mourikis — closed-form VI initialization.
- Mur-Artal & Tardós, *Visual-Inertial Monocular SLAM with Map Reuse*, arXiv:1610.05949. ~15 s to reliable init.
- Campos et al., *Fast and Robust Initialization for Visual-Inertial SLAM*, arXiv:1908.10653. **Consistent init in under 2 s at 5% scale error, converging to 1% after 10 s of VI bundle adjustment.**
- Campos et al., **ORB-SLAM3**, IEEE T-RO 37(6), 2021. Inertial-only MAP over an up-to-scale visual trajectory recovering scale, gravity, biases, velocities; then joint VI BA.

**1% beats every learned prior below, with zero model download.** This is our primary ScaleSource. It is gated entirely on L0.

### Scale via learned priors — the fallback

- Bochkovskii et al., **Depth Pro**, arXiv:2410.02073. Metric depth without intrinsics metadata, plus SOTA single-image focal estimation.
- Piccinelli et al., **UniDepth**, CVPR 2024. Decouples camera prediction from depth via pseudo-spherical representation.
- **MoGe-2**, arXiv:2507.02546. Metric-scale monocular geometry with intrinsics.
- **Learned Monocular Depth Priors in Visual-Inertial Initialization**, arXiv:2204.09171 (ECCV 2022). Upgrades mono-depth to metric by jointly optimizing scale and shift; shows classical VI init is ill-conditioned exactly for the non-gesticulating smartphone user.
- **MDE-VIO**, arXiv:2602.11323. Affine-invariant depth consistency + ordinal constraints + variance gating into a VINS-Mono backend, edge-targeted. **Reports that direct metric depth predictions were insufficient** — use them as constraints, never as truth.

### Place recognition, loop closure, pose graph — mature, port it

- Gálvez-López & Tardós, *Bags of Binary Words for Fast Place Recognition in Image Sequences*, IEEE T-RO 28(5), 2012. **DBoW2** — the practical choice. Binary descriptors, small vocabulary, fast, C++ that compiles cleanly to WASM.
- Mur-Artal & Tardós, **ORB-SLAM2**, IEEE T-RO 33(5), 2017. Reference implementation of keyframe map + relocalization + loop closure, and the lineage AlvaAR already forked.
- ORB-SLAM3 (cited above) improves place-recognition recall and adds map merging.
- Kümmerle et al., **g2o**, ICRA 2011. Pose graph backend. Compiles to WASM; Eigen-based.
- Sarlin et al., *From Coarse to Fine: Robust Hierarchical Localization at Large Scale*, CVPR 2019, and Arandjelović et al., **NetVLAD**, CVPR 2016 — learned place recognition. Better recall, far heavier. Only if DBoW2 recall proves inadequate.

**Critical failure mode:** a false-positive loop closure corrupts the map irrecoverably and is *worse than no loop closure at all*. Geometric verification after every place-recognition hit is non-negotiable, and false-positive rate is a first-class metric (§5), not a footnote.

### Feed-forward geometry — context, not our path

MASt3R-SLAM (CVPR 2025), VGGT-SLAM (arXiv:2505.12549), SLAM-MER (CVPR 2026, MERL TR2026-056), EC3R-SLAM (ICRA 2026). All uncalibrated-capable, all far too heavy for a phone browser. Relevant if we ever add a host-side mode.

### Distillation — technique to borrow if a learned source is needed

- **LingBot-VLA**, arXiv:2601.18692 and **LingBot-VLA 2.0**, arXiv:2607.06403. Learnable query tokens aligned against a frozen teacher's depth tokens via L1 on projected features; teacher discarded at inference. Also distils a *future* query at horizon T — directly applicable as a latency compensator.

**Key insight:** compression comes from narrowing the *domain*, not from int8. A teacher that must handle every scene on earth has capacity we don't need.

### Reference implementation

- **AlvaAR** — github.com/alanross/AlvaAR. OV²SLAM + ORB-SLAM2 through Emscripten. **GPLv3 — verify licence compatibility before forking.** Best existing source for browser frame plumbing.
- **8th Wall** — engine open-sourced MIT, *SLAM excluded*; SLAM available binary-only. Hosted services offline Feb 2027. Read the frame/camera pipeline; do not build on it.

---

## 6. Verification

**This is the section that matters.** Each layer has a different ground truth and must be validated independently. A system-level number tells us nothing about which layer is broken.

### Test tiers and cadence

The per-layer sections below say *what* to measure. This says when it runs and what it needs.

| Tier | Needs | Runtime | Cadence |
|---|---|---|---|
| **1 — Pure** | nothing | < 10 s | every commit |
| **2 — Replay** | datasets, native build | seconds–minutes | subset per commit, full nightly |
| **3 — Rig** | arm / turntable / strobe | minutes | nightly |
| **4 — Browser** | device matrix | slow, flaky | per milestone, pre-release |

**Tier 1 — pure.** Everything with a closed-form answer, on synthetic input. Lie group ops against known identities; homography decomposition on generated correspondences; clock-model recovery from synthetic jitter; covariance propagation on a linear system with an analytic solution. Plus property tests: `SE(3)` exp/log round-trip, map serialise/deserialise round-trip. This tier catches most bugs and must stay fast enough that nobody is tempted to skip it.

**Tier 2 — replay.** EuRoC and TUM-VI through the native build. **This is the regression wall.** Per-sequence ATE checked into `harness/baselines/` as data; CI fails on regression beyond tolerance. Doubles as port validation, since these sequences have published reference numbers.

**Tier 3 — rig.** The arm makes this automatable in a way handheld testing never is. Build a deliberate trajectory set and keep it fixed:

- static hold — degenerate for inertial scale, and should be *detected* rather than silently wrong
- pure rotation — the tier-2 survival case
- slow translation, fast translation
- revisit loop — relocalization and closure
- one adversarial case: low texture or fast rotation

Scale error, NEES and coverage all live here, because they need many trials against truth.

**Tier 4 — browser.** Only this tier catches native-vs-WASM numerical divergence and browser plumbing failures — **keep them separable** by feeding a *recorded* frame source into the browser build, so a failure identifies which of the two it is. Playwright automates Chrome Android; iOS Safari realistically needs a device lab or a human.

### Determinism is a prerequisite

A pipeline that reads wall-clock time and live frames cannot be regression-tested at all. Non-negotiable from M0:

- **No `Date.now()` or `performance.now()` anywhere in the pipeline.** Every timestamp enters through the clock layer.
- **Every RNG is seeded** and the seed is logged. RANSAC included.
- **The frame source is an interface**, with live and replay implementations.

The same binary then runs live *and* replays a canned trajectory bit-for-bit reproducibly. Retrofitting this is miserable; it belongs in M0.

### Log to rerun in CI, not just locally

Nightly runs record a rerun session as a build artifact. When a regression fires, scrub the failing run visually instead of bisecting from a single ATE number.

### Ground-truth rigs (build these first)

| Rig | Provides | Build cost |
|---|---|---|
| **Turntable** — phone on a motor at programmable constant rate | exact angular velocity | ~1 day |
| **Strobe** — second display flashing at a known frequency incommensurate with capture rate | absolute camera frame timing | ~1 day |
| **Robot arm** — phone mounted to SO-101 end effector, driven through scripted trajectories | **exact metric 6-DoF from joint encoders + FK** | ~2 days, hardware on hand |
| **ChArUco board** | per-device intrinsics ground truth (validation only, never runtime) | hours |

The robot arm rig is the highest-value item here: repeatable, programmable, unlimited trials, and metrically exact. It converts scale validation from a judgement call into a measurement.

### L0 — Clock

- **Method:** phone on the turntable, viewing the strobe. Cross-correlate gyro angular rate against image-derived rotation rate; the peak lag *is* the offset. Strobe gives independent absolute camera timing.
- **Metrics:** residual offset after correction (ms); **variance** of residual offset — the jitter is the thing we claim to fix, so report the distribution, not a mean.
- **Bar:** beat the 30 ms native-access figure from arXiv:2001.00470. If we cannot, say so publicly and downgrade the inertial ScaleSource.
- **Report per device.** Aggregating hides the failure cases.

### L1 — Orientation

- Turntable at known rate → integrated angle error over 60 s (deg).
- Static on a level surface → roll/pitch error vs gravity (deg).
- Yaw drift rate (deg/min), vision disabled.

### L2 — Intrinsics

- **Ground truth:** ChArUco calibration per device.
- **Metric:** relative focal error `|f̂ − f|/f`, distribution over ≥30 trials × ≥6 devices.
- **Required ablations**, because the literature says both are failure modes:
  - with / without radial distortion in the model
  - with / without lever-arm (camera-IMU translation) modelling
  - across scene depth (barrel distortion and lever-arm parallax interact with depth)

### L3 — Tracking

- **Replay validation:** feed EuRoC and TUM-VI through the WASM build. Compare against the native reference on identical sequences.
- **Metric:** ATE after Sim(3) alignment (scale-free — L3 does not claim scale). Should match native within noise. **Any divergence is a port bug, not an algorithm result.**
- **Separate plumbing from algorithm:** run the same sequences in-browser on real phones via a mocked frame source. Divergence between desktop-WASM and phone-browser isolates browser plumbing.

### L4 — Map

**Relocalization**
- Protocol on the arm rig: build a map, drive the arm away, occlude the camera, return via a *different* path, measure recovery.
- Metrics: success rate as a function of viewpoint change (baseline distance and view angle); time-to-relocalize; pose error immediately post-relocalization vs ground truth.
- Also run 7-Scenes for a comparable public number.

**Loop closure**
- ATE on EuRoC with and without loop closure enabled — the standard comparison, and it validates the port.
- **False-positive rate of place recognition, measured separately and reported prominently.** Sample non-revisited sequences and count spurious matches surviving geometric verification. Target zero; anything above zero needs the verification threshold raised until it is.

**Persistence and scale retention**
- The money metric for the map-as-ruler claim: anchor a map with a known-good ScaleSource, serialise it, relocalize into it in a fresh session, measure scale error against the arm.
- Report degradation across session gaps and across lighting change.

**Resource envelope**
- Map memory growth vs session duration (MB/min) — a phone tab will be killed if this is unbounded.
- Backend latency distribution, and confirmation the frontend never stalls on it. Measure frame-time tail (p99), not mean.

### L5 — Scale (the headline metric)

- **Ground truth:** robot arm rig. Phone on the end effector, scripted trajectories spanning the excitation spectrum — from near-static (worst case for inertial) to vigorous.
- **Primary metric:** scale error % as a function of time-since-init. **Directly reproduce the Campos curve**: report our 2 s and 10 s numbers against their 5% / 1%.
- **Paired design:** every ScaleSource evaluated on *identical recorded trajectories*, so comparisons are within-trajectory. Reduces variance substantially over independent runs.
- **Report the excitation dependence explicitly.** A single aggregate number for inertial scale is misleading, because the theory says accuracy collapses as translational acceleration vanishes. Plot error against measured excitation.

### L6 — Uncertainty calibration

We claim the covariance is meaningful. Prove it.

- **NEES** (normalised estimation error squared) over ≥100 trials against arm ground truth. Should track chi-squared with the state dimension. Consistently above the bound = overconfident, which is worse than no covariance at all.
- **Coverage:** do 95% intervals contain truth 95% of the time? Report empirical coverage at 68/95/99.
- This is standard estimation practice and essentially absent from shipping AR SDKs. It is our clearest differentiator and it is cheap — the estimator already computes it.

### System level

- **Device × OS matrix.** iOS 26+ (WebGPU floor) and Android Chrome. Report per cell, never pooled — device-specific quirks were 8th Wall's actual moat and will be our actual bug surface.
- **Thermal:** 15-minute sustained runs logging pose error and frame rate against elapsed time. Report time-to-degradation per device.
- **Rolling shutter:** sweep arm velocity, plot error against angular rate. Establishes and documents the motion envelope rather than pretending it doesn't exist.

### Statistical design

- Fix the primary metric (scale error at 10 s) and pre-register it before running anything.
- Power-analyse trial counts for the effect sizes we care about — a 1% vs 2% scale difference needs a specific N; guessing wastes weeks.
- Paired trials wherever conditions can share a trajectory.
- Correct for multiple comparisons across the ScaleSource × device grid.

---

## 7. Implementation and toolchain

**Rust, compiled to `wasm32-unknown-unknown` via `wasm-bindgen`.** Not C++/Emscripten. Three reasons, in order of weight:

**1. `wgpu` gives us one GPU codebase, native and web.** `wgpu` is the Rust implementation of WebGPU and targets Metal/Vulkan/DX12 natively as well as the browser. L3's compute shaders are written once in WGSL and run in both. This is what makes §5 tractable: the EuRoC replay harness, the arm-rig regression suite and the NEES computation all run as native `cargo test` at full speed, executing the same shader code that ships to the browser. Browser-only testing of a numerical pipeline is intolerable.

**2. It resolves the AlvaAR licence risk.** Rust means clean-room by construction. GPLv3 contamination stops being a decision we have to make.

**3. Binary size and reproducibility.** `wasm-opt` output is materially smaller than Emscripten's for a library others will embed. Cargo over CMake-plus-Emscripten for build reproducibility.

### The cost, stated plainly

The prior art in §4 is entirely C++ — ORB-SLAM2/3, DBoW2, g2o, Ceres, Eigen. We fork none of it. Two things make this survivable:

- **DBoW2 is small.** The vocabulary is the artifact; the code is a tree search over binary descriptors. Reimplementable in days, and the trained vocabulary file is reusable as data.
- **We need a pose-graph optimizer, not a solver framework.** Gauss-Newton over SE(3) with `nalgebra` + `sophus-rs`, or `factrs` / `tiny-solver` off the shelf. g2o and Ceres are general; our problem is not.

Budget the reimplementation explicitly in M4a/M4b rather than discovering it.

### Crate selection

| Need | Crate |
|---|---|
| Linear algebra | `nalgebra` |
| Lie groups (SE(3)/SO(3)) | `sophus-rs` |
| GPU compute | `wgpu` + WGSL |
| Pose graph / factor graph | `factrs` or `tiny-solver`; hand-rolled GN as fallback |
| JS bindings | `wasm-bindgen` |
| Threads (if COOP/COEP accepted) | `wasm-bindgen-rayon` |

Evaluate the solver crates against a known pose-graph benchmark in M0 before committing.

### Repository shape

Cargo workspace. **The layer boundaries from §4 are crate boundaries, so the compiler enforces them** — L3 cannot accidentally reach into L4.

```
web-slam/
├── Cargo.toml                  # workspace root
├── crates/
│   ├── wslam-core/             # Pose, SE3, covariance, TrackingState, FrameSource trait
│   │                           #   depends on nothing else in the workspace
│   ├── wslam-gpu/              # wgpu device setup + WGSL kernels
│   ├── wslam-clock/            # L0   (feature-gated: tight-vi)
│   ├── wslam-orientation/      # L1
│   ├── wslam-calib/            # L2
│   ├── wslam-track/            # L3   frontend; uses wslam-gpu
│   ├── wslam-map/              # L4   keyframes, place recognition, pose graph
│   ├── wslam-scale/            # L5   ScaleSource trait + implementations
│   ├── wslam/                  # orchestration — the ONLY crate aware of all layers.
│   │                           #   Owns threading and the frontend/backend split.
│   └── wslam-wasm/             # wasm-bindgen boundary. Deliberately thin —
│                               #   no logic, so the core stays natively testable.
├── packages/
│   ├── web-slam/               # npm package: TS shim + wasm artifact + types
│   └── demo/                   # public demo site (M0 stub → M7 real)
├── harness/
│   ├── replay/                 # native replay binary, EuRoC / TUM-VI
│   ├── viewer/                 # rerun logging helpers
│   └── baselines/              # checked-in ATE + scale regression data
├── rigs/                       # Python: arm trajectories (LeRobot), turntable, strobe
├── vocab/                      # DBoW vocabulary artifact (git-lfs)
├── datasets/                   # gitignored; fetch script
└── xtask/                      # cargo xtask: build-wasm, bench, regen-baselines
```

**Dependency direction is one-way and enforced.** `wslam-core` depends on nothing; every layer crate depends only on `wslam-core` (plus `wslam-gpu` where relevant); `wslam` composes them. If a layer needs to reach sideways, that's a design smell the build will surface immediately.

**Feature flags gate the sensor tiers**, so tier-3 code doesn't ship to tier-2 consumers:

```toml
[features]
default = ["orientation", "map"]
orientation  = []                    # tier 2
tight-vi     = ["wslam-clock"]       # tier 3, research track R1
learned-scale = []                   # pulls model loading + weights
```

**The TS shim lives in `packages/web-slam/` and is reviewed under the §3 rule** — it moves bytes and timestamps, nothing else.

`xtask` rather than Makefiles, keeping the build in Rust and cross-platform for a team that will be on mixed machines.

### The TypeScript shim

Browser sensor APIs are reachable only from JS, so a thin TS layer around `getUserMedia`, `requestVideoFrameCallback` and `DeviceMotion` is unavoidable.

**Keep it deliberately stupid.** It stamps arrival at the earliest possible instant and passes raw bytes and raw timestamps into WASM. No buffering, no smoothing, no reordering, no processing of any kind. L0 exists to measure event-loop jitter; a shim that helpfully cleans up its inputs destroys the signal it is supposed to deliver. This is a review rule, not a style preference.

Everything above the shim — clock modelling, orientation, intrinsics, tracking, mapping, scale, uncertainty — is Rust.

### Unchanged by this decision

Threading still requires `SharedArrayBuffer` and therefore COOP/COEP headers. See §3 L4 and the risk table. Rust does not help here.

---

## 8. Milestones

**Baseline configuration is vision + loose orientation.** Tight visual-inertial fusion is a parallel research track that gates nothing.

| # | Deliverable | Exit criterion |
|---|---|---|
| M0 | Harness, replay pipeline, ground-truth rigs, dev viewer, **API frozen against a mock** | EuRoC replays end-to-end; arm produces logged GT; viewer shows features/landmarks/trajectory live and in replay; public demo runs end-to-end on a stub backend |
| M1 | L1 loose orientation + L2 intrinsics | Focal error distribution with both ablations; survives pure-rotation segments without session loss |
| M2 | L3 tracker | ATE matches native reference on EuRoC within noise |
| M3 | L4a keyframe map + relocalization | Relocalization success-rate curve vs viewpoint change; recovery under 1 s |
| M4 | L5 ScaleSource — `fiducial`, `declared`, `map` | Scale error vs arm ground truth, per source, paired trajectories |
| M5 | L4b loop closure + pose graph | EuRoC ATE improves with closure enabled; **zero** verified false positives |
| M6 | L6 uncertainty calibration | NEES within chi-squared bounds; coverage within 2% of nominal |
| M7 | L4c persistence + **public demo** | Scale retained across sessions; demo runs on stock iOS Safari and Android Chrome |
| **R1** | L0 clock + tight VI + `inertial` scale | *Parallel research track.* Residual offset variance beaten; Campos curve reproduced. Failure costs an upgrade, not the product. |

**M2 is the critical path**, not the clock layer. If the tracker port doesn't match its native reference, nothing above it is trustworthy.

### Visualization

Two separate artifacts with different audiences. Do not conflate them.

**Dev viewer — M0, internal, ugly, indispensable.** Use [rerun](https://rerun.io) for the native loop: Rust-native spatial logging, log directly from `cargo test`, gives point clouds, camera frusta, time-scrubbing and plots without writing a viewer. Must show:

- Camera feed with tracked features overlaid, coloured by state — new / tracked / outlier-rejected / lost
- Sparse landmark cloud and keyframe frusta in 3D
- Estimated trajectory overlaid on ground truth (arm rig or EuRoC)
- **Covariance ellipsoid on the current pose** — it is our differentiator, we should be looking at it daily
- Scale source badge and current scale variance
- Per-stage frame timing: upload / pyramid / corners / flow / PnP. Required to manage the WebGPU budget.
- Tracking state machine: init / tracking / lost / relocalizing
- From M5: pose-graph edges **including loop candidates rejected by geometric verification** — this is how the false-positive threshold gets tuned by eye rather than by guesswork

**Public demo — M7, browser, polished.** A single page, no install, QR-scannable. Live camera with the landmark cloud and trajectory rendered over it, keyframe frusta accumulating as you walk, a visible relocalization event when you cover and uncover the lens. Three.js is fine here; it is decoupled from the compute path and not performance-critical.

The demo is also the adoption artifact. Nobody adopts a tracking library from a README.

---

## 9. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Browser clock jitter irreducible | Inertial scale unviable | Discover at M1; fall back to other sources |
| Barrel distortion breaks focal init | L2 unusable | Distortion in model; ablation is a gate, not a nice-to-have |
| Wrist lever arm violates pure rotation | Focal bias | Estimate as camera-IMU extrinsic |
| WebGPU requires iOS 26 | Excludes older iPhones | Documented floor; no WebGL2 fallback planned |
| Rolling shutter under fast motion | Error under vigorous handheld use | Characterise and publish the envelope |
| Thermal throttling | Long-session degradation | Measure and document; not solvable |
| **COOP/COEP required for threads** | Breaks third-party embedding — often the whole point of WebAR | Decide before M4: single-threaded backend vs header requirement |
| **False-positive loop closure** | Irrecoverable map corruption, worse than no closure | Mandatory geometric verification; FP rate is a release gate |
| Unbounded map memory | Tab killed on long sessions | Keyframe culling; measure MB/min from M4a |
| ~~AlvaAR GPLv3~~ | Resolved | Rust clean-room removes the licence question |
| **C++ prior art not forkable** | DBoW2 and pose-graph reimplementation cost | Budgeted explicitly in M4a/M4b; solver crates evaluated at M0 |

---

## 10. Open decisions

**Which ScaleSources are we required to support?** This determines a large fraction of the work.

If `declared` (one user tap on a known distance) is acceptable to our consumers, most of the difficulty evaporates — it is exact, free, and needs no compute. If we must be fully passive, M1 becomes load-bearing and the timeline roughly doubles.

Decide before M0.

**Do we accept the COOP/COEP header requirement?** Threads make the L4 backend comfortable but block third-party embedding, which is frequently the entire reason a team chooses WebAR over native. The alternative is a single-threaded backend running loop closure at a reduced rate.

Decide before M4.
