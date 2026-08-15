# web-slam

**A 6-DoF pose source for stock mobile browsers, with an explicit metric anchor
and calibrated uncertainty.**

30–60 Hz camera pose inside an unmodified mobile browser — including iOS Safari
— from `getUserMedia` and `DeviceMotion` alone. No app, no WebXR, no plugin.

```ts
import { WebSlam } from 'web-slam';

const slam = await WebSlam.create({ video: videoEl });
startButton.onclick = () => slam.start();

renderLoop(() => {
  const pose = slam.currentPose();
  if (pose) camera.matrix.fromArray(pose.matrix);
});
```

Two properties distinguish this from every other option:

1. **Metric scale comes from a declared, pluggable source.** The library never
   silently guesses scale. You choose an anchor and accept its tradeoffs.
2. **Every pose carries a covariance**, and that covariance is validated to be
   statistically calibrated rather than decorative.

The full design rationale is in [`spec.md`](./spec.md). Read it before changing
anything structural — most of the surprising decisions here are load-bearing and
the spec says why.

---

## Why this exists

WebXR is unavailable on iOS. Safari implements no `immersive-ar`, and because
every iOS browser is WebKit underneath, no browser install fixes it. This is the
single reason browser-based world tracking is still an open problem.

And monocular metric scale is *unobservable* — a theorem, not a gap in the
literature. A scene twice as large at twice the distance produces pixel-identical
images. So scale always comes from a ruler, and there are only a few:

| Ruler | Accuracy | Cost | Status |
|---|---|---|---|
| `none` | — | free | **default**, honest |
| `declared` — one user tap on a known distance | exact | one interaction | shipped |
| `fiducial` — marker of known size | exact | object must be visible | shipped |
| `map` — relocalize into an anchored map | inherits its anchor | must relocalize | shipped |
| `learned` — monocular depth prior | several % | model download, GPU | opt-in |
| `inertial` — double-integrated acceleration | ~1% given excitation | tight time sync | research track R1 |

Most systems pick one and hide it. This one makes you say which.

---

## Repository layout

Layer boundaries from spec.md §4 are **crate boundaries**, so the compiler
enforces them — L3 cannot accidentally reach into L4.

```
crates/
  wslam-core/          Pose, SE(3), covariance, TrackingState, FrameSource
                       — depends on nothing else in the workspace
  wslam-gpu/           wgpu device setup + WGSL kernels
  wslam-clock/         L0  unified timebase          (feature: tight-vi)
  wslam-orientation/   L1  gyro + gravity
  wslam-calib/         L2  focal length from rotation
  wslam-track/         L3  KLT + PnP frontend
  wslam-map/           L4  keyframes, place recognition, pose graph
  wslam-scale/         L5  ScaleSource trait + implementations
  wslam/               orchestration — the only crate aware of all layers
  wslam-wasm/          wasm-bindgen boundary, deliberately thin
packages/
  web-slam/            npm package: TS shim + wasm artifact + types
  demo/                the public demo
harness/
  replay/              native replay: EuRoC, ATE, NEES
  viewer/              rerun logging helpers
  baselines/           checked-in regression data
rigs/                  Python: turntable, strobe, ChArUco (no robot arm — see rigs/README.md)
vocab/                 DBoW vocabulary artifact
xtask/                 cargo xtask: build-wasm, test, regen-baselines
docs/                  ARCHITECTURE, CONTRACT, DECISIONS, VERIFICATION
```

Dependency direction is one-way. If a layer needs to reach sideways, the build
surfaces it immediately.

---

## Getting started

```sh
# Rust side
cargo test --workspace              # Tier 1: pure, synthetic, < 10 s
cargo xtask ci                      # everything CI runs

# TypeScript side
pnpm install
pnpm -r test
pnpm --filter web-slam-demo dev     # the demo, against the stub backend

# Build the wasm artifact
cargo install wasm-bindgen-cli
cargo xtask build-wasm --release
```

Requires Rust 1.82+, Node 22+, and pnpm 10.

---

## Sensor tiers

Sensor use is declared configuration, not an assumption — the same discipline
applied to scale.

| Tier | Sensors | Needs L0? | Status |
|---|---|---|---|
| 1 | Vision only | no | automatic fallback when motion permission is denied |
| **2** | **Vision + loose orientation** | **no** | **baseline — what we ship** |
| 3 | Tight visual-inertial | yes | optional; research track R1 |

Tier 2 gets gravity direction, an orientation prior, and survival through brief
vision failure — none of which need sub-frame temporal alignment. The only thing
tight coupling adds is inertial metric scale, and scale is already pluggable.
That is why L0 is off the critical path.

---

## Verification

Each layer has a different ground truth and is validated independently. A
system-level number tells you nothing about which layer is broken.

| Tier | Needs | Runtime | Cadence |
|---|---|---|---|
| 1 — Pure | nothing | < 10 s | every commit |
| 2 — Replay | datasets | seconds–minutes | subset per commit, full nightly |
| 3 — Rig | turntable / strobe | minutes | manual |
| 4 — Browser | device matrix | slow, flaky | per milestone |

Three rules are structural rather than aspirational, and CI checks them:

- **No wall clock in the pipeline.** Every timestamp enters through
  `wslam_core::TimeBase`. `cargo xtask check-invariants` enforces it, and runs in
  CI — it is a Rust checker rather than a grep because a grep cannot see comments
  or module boundaries, fires on doc comments and on legitimate test stopwatches,
  and a check with false positives gets disabled within a week.
- **Every RNG is seeded**, RANSAC included. `DeterministicRng` has no
  `from_entropy` constructor, and the same checker rejects `thread_rng` and
  friends anywhere in `crates/` or `harness/`.
- **The frame source is an interface.** The same binary runs live and replays a
  canned trajectory bit-for-bit reproducibly.

See [`docs/VERIFICATION.md`](./docs/VERIFICATION.md) for what each layer
measures and against what.

### The uncertainty claim, specifically

We claim the covariance is meaningful, so we test it:
`wslam_core::covariance::ConsistencyAccumulator` computes NEES against
chi-squared bounds and empirical coverage at 68/95/99. An estimator that is
*overconfident* is flagged separately from one that is merely conservative,
because overconfidence is worse than no covariance at all.

This is standard estimation practice and essentially absent from shipping AR
SDKs. It is the clearest differentiator and it is cheap — the estimator already
computes it.

---

## Non-goals

Explicitly out of scope, and we resist all of them:

- Dense mapping or reconstruction — the map is sparse keyframes and landmarks
- Plane detection, image targets, face tracking, occlusion, rendering
- Multi-user / cloud-shared maps
- Matching ARKit/ARCore accuracy
- Working on every consumer phone in existence

That last one is what consumed 8th Wall for seven years. This is a component,
not a platform.

## Known limits

Documented rather than hidden:

- **WebGPU requires iOS 26.** No WebGL2 fallback is planned. Older iPhones are
  out of scope.
- **Rolling shutter** degrades accuracy under fast motion. The envelope is
  characterised and published rather than pretended away.
- **Thermal throttling** degrades long sessions. Measured and documented; not
  solvable from here.
- **Inertial scale is unobservable during a static hold.** The `inertial` source
  returns no estimate rather than a confident wrong one.
- **No fisheye camera model.** Brown-Conrady radial-tangential only, which rules
  out TUM-VI and any ultra-wide lens that needs Kannala-Brandt.

---

## Licence

MIT OR Apache-2.0. Clean-room Rust throughout; no GPL code is read, vendored or
referenced. See [`docs/DECISIONS.md`](./docs/DECISIONS.md) D6.

---

## Status

Every gate below is green as of the initial build. `cargo xtask ci` runs them all.

| Check | Result |
|---|---|
| `cargo test --workspace --all-features` | **774 passed, 0 failed** |
| `packages/web-slam` API conformance (vitest) | 34 passed |
| `rigs` Tier-1 instrument tests (pytest) | 8 passed |
| `cargo clippy --all-targets --all-features -- -D warnings` | clean |
| `cargo fmt --all --check` | clean |
| `cargo xtask check-invariants` | no wall clock on the estimation path; no unseeded RNG |
| `cargo check -p wslam-wasm --target wasm32-unknown-unknown` | builds |
| `packages/demo` production build | builds |

816 tests total. Layers L0–L6, the orchestrator, the wasm boundary, the npm
package, the demo and the replay harness are all implemented and covered.

**What is measured on real data**, stated plainly because the project's
thesis is that its numbers can be trusted:

- **L3 ATE, GPU front-end, EuRoC `MH_01_easy`: 0.309 m RMSE at 100% coverage,
  0.1% frame loss.** Down from the 3.57 m headline this section used to carry.
  What moved it: the GPU image front-end (also ~15× faster than the CPU
  reference at 1.3 ms/frame median), relocalization that actually fires
  (covisibility-expanded matching plus a projection-guided second pass — the
  old matcher starved one or two correspondences short of the gate on every
  real attempt), Sim(3) epoch merging when place recognition proves a
  post-loss segment overlaps an older one (ORB-SLAM3's map merge, solved by
  RANSAC-robust Umeyama on matched landmark pairs), and an orientation prior
  that is gated and per-frame arbitrated instead of trusted (the ungated
  prior was *costing* accuracy; see `docs/VERIFICATION.md`).
  Still 3–20× above the 0.016–0.100 m published band — the remaining gap is
  within-segment drift, and the honest next step is a Sim(3)-aware backend,
  not parameter tuning (every knob combination that helped one sequence hurt
  the others; the measurements are in the doc).
  **Use EuRoC, not TUM-VI:** TUM-VI is fisheye (Kannala-Brandt) and this build
  implements Brown-Conrady only, so the loader refuses it rather than misreading
  the coefficients. See `datasets/README.md`.
- **Loop closure verifies but contributes no pose-graph edges.** Measured on
  MH_01: every edge weighting tried made the corrected trajectory *worse* than
  the raw one, because at 0.3 m of drift the closure's own error — landmark
  drift the PnP covariance cannot see, plus monocular scale drift an SE(3)
  edge cannot express — exceeds the drift it could fix. A verified loop's
  value today is what it proves (same place, both frames), which is what the
  epoch merge consumes. The edges come back when the graph speaks Sim(3).
- **L5 scale error %** and **L6 NEES on real data** have no metric ground-truth
  source, because the robot-arm rig is out of scope. Both are validated
  synthetically. See `rigs/README.md` and `docs/VERIFICATION.md`.
