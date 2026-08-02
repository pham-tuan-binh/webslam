# Architecture

Six layers. Layers 0–3 are scene-agnostic and reusable. Layer 4 is where domain
assumptions enter. Layer 5 is the only opinionated one.

```
L6  Output          pose + covariance + scale provenance
L5  ScaleSource     ← pluggable, the only opinionated layer
L4  Map             keyframes, place recognition, pose graph  ← async backend
L3  Tracking        sparse KLT + PnP frontend, pose up to scale
L2  Intrinsics      focal length, estimated once at init
L1  Orientation     gyro + gravity, drift-free in roll/pitch
L0  Clock           unified timebase for camera and IMU
```

**The layer boundaries are crate boundaries**, so the compiler enforces them.
That is not stylistic: a tracker that can reach into the map will eventually
reach into the map, and then the frontend blocks on the backend and the p99
frame time is nobody's fault in particular.

```
wslam-core        → (nothing in-workspace)
wslam-gpu         → core
wslam-clock       → core                  L0, feature-gated: tight-vi
wslam-orientation → core                  L1
wslam-calib       → core                  L2
wslam-track       → core, gpu             L3
wslam-map         → core                  L4
wslam-scale       → core, map             L5
wslam             → all of the above      orchestration
wslam-wasm        → wslam, core           the JS boundary
```

---

## The two seams that matter

### 1. Frontend / backend

L3 runs every frame on the critical path. L4 does not. **The frontend must never
block on the backend.**

Under the default single-threaded build (docs/DECISIONS.md D2) this is enforced
by a time-sliced budget rather than by a thread boundary: the orchestrator gives
the backend `MapConfig::backend_budget_ms` per frame and it yields when the
budget is spent, resuming next frame. Relocalization — a BoW query plus one PnP
— is fast enough to run inline, which matters because the user is standing there
waiting. Pose-graph optimisation is the slow part, and it is also the part that
tolerates latency best.

The measurable consequence is a p99 frame-time claim, not a mean, and
spec.md §6 L4 already requires reporting it.

### 2. The wasm boundary

`wslam-wasm` moves bytes and nothing else. No logic lives there, so everything
above it stays natively testable at full speed — which is what makes the whole
verification plan tractable. Browser-only testing of a numerical pipeline is
intolerable.

The TypeScript shim on the other side follows the same rule, and it is a review
rule rather than a preference: it stamps arrival at the earliest possible
instant and passes raw bytes and raw timestamps through. **L0 exists to measure
event-loop jitter; a shim that helpfully cleans up its inputs destroys the
signal it is supposed to deliver.**

---

## Data flow, one frame

```
   browser                    wslam-wasm            wslam (orchestration)
┌──────────────┐            ┌────────────┐        ┌──────────────────────┐
│ rVFC         │  rgba +    │            │ Frame  │ TimeBase.map_camera  │
│ mediaTime    │──mediaTime→│ pack/unpack│───────→│         ↓            │
│              │            │  only      │        │ L2 intrinsics (init) │
│ devicemotion │  raw deg/s │            │ Imu    │         ↓            │
│ + arrival ms │───────────→│            │───────→│ L1 orientation       │
└──────────────┘            └────────────┘        │         ↓ prior      │
                                                  │ L3 track → pose,cov  │
                                                  │         ↓            │
                                                  │ L4 keyframe? reloc?  │
                                                  │         ↓            │
                                                  │ L5 scale.estimate()  │
                                                  │         ↓            │
                                                  │ L6 Pose + cov + prov │
                                                  └──────────────────────┘
```

Two things travel with the pose all the way out, and are never available by a
separate query: the **covariance** and the **scale provenance**. Separate
queries get skipped.

---

## Where uncertainty comes from

Each stage contributes, and the composition is explicit rather than a tuned
constant:

| Stage | Contribution |
|---|---|
| L3 PnP | `(JᵀΣ⁻¹J)⁻¹` at the solution, in the `[translation, rotation]` right-perturbation convention |
| L4 relocalization | Verification inlier geometry, transported through the map pose |
| L5 scale | `Var(s·t) = s²Var(t) + t²Var(s)` — the second term is the one usually dropped, and dropping it is the standard way an estimator becomes overconfident |
| L4 map anchor | `anchor.inflated_by(reloc_variance)` — a map may never report itself as more certain than its origin |

`Pose::with_scale` implements the third row and a Tier-1 test pins it.

---

## Sensor tiers

| Tier | Sensors | Needs L0? | What happens |
|---|---|---|---|
| 1 | Vision only | no | Automatic fallback when motion permission is denied. Tracking still works; there is no orientation prior and no gravity direction. |
| **2** | **Vision + loose orientation** | **no** | The baseline. Gravity direction, an orientation prior for prediction, survival through brief vision failure. |
| 3 | Tight visual-inertial | yes | Adds inertial metric scale, and only that. |

L0 is off the critical path because the *only* thing tight coupling adds is
inertial scale — and scale is already pluggable. Tier 2 needs approximate
alignment, not sub-frame alignment.

Feature flags gate the tiers so tier-3 code does not ship to tier-2 consumers.

---

## What is deliberately absent

- **Dense reconstruction.** The map is sparse keyframes and landmarks. Nothing
  renderable.
- **A rendering layer.** `web-slam/three` is a camera-sync helper, ~150 lines,
  with three.js as a peer dependency it never imports.
- **A plane/anchor/hit-test API.** Out of scope; see spec.md §1.
- **A WebGL2 fallback.** WebGPU is the floor. iOS 26+.
- **Threads by default.** See docs/DECISIONS.md D2.

Each absence is load-bearing. spec.md §1: *"That last one is what consumed 8th
Wall for seven years. We are building a component, not a platform."*
