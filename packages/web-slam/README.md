# web-slam

**6-DoF camera pose in a stock mobile browser — including iOS Safari — with an
explicit metric anchor and calibrated uncertainty.**

No app, no WebXR, no plugin. `getUserMedia` and `DeviceMotion` are the only
inputs.

```sh
npm install web-slam
```

```ts
import { WebSlam } from 'web-slam';

const slam = await WebSlam.create({ video: videoEl });

// iOS requires a user gesture before motion sensors. That is a hard platform
// constraint, so start() is explicit and must be called from a click handler.
startButton.onclick = () => slam.start();

renderLoop(() => {
  const pose = slam.currentPose();
  if (pose) camera.matrix.fromArray(pose.matrix);
});
```

That is the whole default path. Everything below is progressive disclosure.

---

## Scale is declared, never guessed

Monocular metric scale is *unobservable* — a theorem, not a gap. A scene twice
as large at twice the distance produces pixel-identical images. So scale always
comes from a ruler, and this library makes you name yours:

```ts
import { ScaleSource } from 'web-slam';

await WebSlam.create({
  video: videoEl,
  scale: ScaleSource.fiducial({ family: 'apriltag36h11', sizeMeters: 0.1 }),
});
```

| Source | Accuracy | What it costs you |
|---|---|---|
| `ScaleSource.none()` | — | **Default.** Positions are up to scale, and `scale.variance` is `Infinity`. |
| `ScaleSource.declared({ distanceMeters })` | exact | One user tap on a known distance. |
| `ScaleSource.fiducial({ family, sizeMeters })` | exact | A marker must be visible. |
| `ScaleSource.map(savedMap)` | inherits its anchor | Must relocalize first. |
| `ScaleSource.learned({ model })` | several % | Downloads weights. |
| `ScaleSource.inertial()` | ~1% given motion | Requires tier 3; **throws** if unavailable. |

The default is `none`, and it is not a degraded mode — a renderer that only
needs relative camera motion is correctly served by it. What the library will
never do is hand you a number and let you assume it is metres.

`ScaleSource.map()` always reports a **larger** variance than the anchor it
inherited, because relocalization adds error. It will not claim to be more
certain than its origin.

---

## Every pose carries its uncertainty

```ts
interface Pose {
  timestamp: number;        // performance.now() domain
  position: Vec3;           // metres only when scale.source !== 'none'
  rotation: Quaternion;
  matrix: Float32Array;     // 4x4 column-major, renderer-ready
  covariance: Float64Array; // 6x6, [translation, rotation]
  scale: { source: ScaleKind; variance: number };
  state: TrackingState;
  initAgeMs: number;
}
```

Covariance and provenance travel *with* the pose rather than behind a separate
query, because separate queries get skipped. And the covariance is validated to
be statistically calibrated — NEES against chi-squared bounds, empirical
coverage at 68/95/99 — not decorative.

---

## Tracking state

```ts
type TrackingState =
  | 'initializing'
  | 'tracking'
  | { limited: 'excessive-motion' | 'insufficient-features' | 'low-light' }
  | 'relocalizing'
  | 'lost';
```

Tracking loss is **not** an edge case on a phone — occlusion, pocketing and fast
motion make it routine. Handle `relocalizing`:

```ts
import { hasPose, isLimited } from 'web-slam';

slam.onState((state, prev) => {
  if (state === 'relocalizing') showHint('Point at somewhere you have been');
  if (isLimited(state)) showHint(`Tracking degraded: ${state.limited}`);
});

slam.onRelocalize(({ mapPoseId }) => hideHint());
```

---

## Pull vs push

Both, and the distinction is deliberate:

- **`slam.currentPose()`** — pull. For renderers: you want the freshest pose at
  *your* draw time, not at every camera frame.
- **`slam.onPose(cb)`** — push, once per tracked frame. For recorders, teleop
  transmitters, anything that must not drop samples.

---

## Map persistence

```ts
const bytes = await slam.saveMap();      // you store it
localStorage.setItem('map', /* ... */);

const slam2 = await WebSlam.create({
  video: videoEl,
  scale: ScaleSource.map(bytes),         // relocalizes into metric
});
```

Anchor scale once, persist the map, and every later session recovers metric by
relocalizing.

---

## three.js

```ts
import { CameraSync } from 'web-slam/three';

const sync = new CameraSync(camera);
renderLoop(() => {
  sync.update(slam.currentPose());
  renderer.render(scene, camera);
});
```

`three` is a peer dependency and is never imported by this package — the helper
is typed structurally against the two properties it touches.

**Set your FOV from the estimated intrinsics.** Rendering with three's default
50° over a 66° camera feed produces a registration error that looks exactly like
a tracking bug, and is the most common integration mistake:

```ts
sync.setIntrinsics(focalPx, video.videoWidth, video.videoHeight);
```

---

## Debug surface

```ts
slam.debug.landmarks();   // Float32Array xyz
slam.debug.keyframes();
slam.debug.trajectory();
slam.debug.features();    // 2D, with per-feature state for overlay colouring
slam.debug.poseGraph();   // edges, including REJECTED loop candidates
slam.debug.timings();     // per-stage: upload / pyramid / corners / flow / pnp
```

**Explicitly unstable.** Versioned separately via `DEBUG_VERSION` so the viewer
can move fast without pinning the core API. Pin an exact version if you build on
it.

---

## Requirements and limits

- **WebGPU**, therefore **iOS 26+** and a recent Android Chrome. No WebGL2
  fallback is planned.
- **A user gesture** before `start()`, for motion permission on iOS.
- **A secure context** (`https`, or `localhost`) for `getUserMedia`.

Denied motion permission is not fatal: the session drops to sensor tier 1
(vision only) and keeps tracking, without a gravity direction or an orientation
prior.

Documented rather than hidden:

- **Rolling shutter** degrades accuracy under fast motion.
- **Thermal throttling** degrades long sessions.
- **Inertial scale is unobservable during a static hold**, so that source
  returns no estimate rather than a confident wrong one.

---

## Determinism

Every RNG in the pipeline is seeded, RANSAC included, and no wall clock is read
on the estimation path. Pass `seed` to make a session reproducible:

```ts
await WebSlam.create({ video: videoEl, seed: 1234 });
```

## Licence

MIT OR Apache-2.0.
