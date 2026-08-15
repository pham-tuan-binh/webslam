/**
 * The public demo.
 *
 * spec.md §3: *"The public demo is written at M0 against a stub implementation
 * that returns synthetic poses ... The point is to validate the API with a real
 * consumer before any of it exists, and to make the interface expensive to
 * change afterwards rather than cheap to drift."*
 *
 * Read this file as a review of the API, not just as a demo. Every awkwardness
 * here is an API bug. Notably, the whole render path is:
 *
 *   const pose = slam.currentPose();
 *   if (pose) sync.update(pose);
 *
 * — which is the one-screen default path spec.md §3 promises. Everything else
 * on screen comes from the explicitly-unstable debug surface.
 *
 * Swapping to the real backend is a two-line change; see `createBackend`.
 */

import { ScaleSource, WebSlam, isLimited, type Pose, type TrackingState } from 'web-slam';
import { CameraSync } from 'web-slam/three';
import { SceneView } from './scene.js';

const el = <T extends HTMLElement>(id: string): T => {
  const node = document.getElementById(id);
  if (!node) throw new Error(`demo: missing #${id}`);
  return node as T;
};

const video = el<HTMLVideoElement>('feed');
const canvas = el<HTMLCanvasElement>('scene');
const gate = el('gate');
const startButton = el<HTMLButtonElement>('start');
const gateError = el('gate-error');
const trailLength = el('trail-length');
const stateDot = el('state-dot');
const stateLabel = el('state-label');
const scaleBadge = el('scale-badge');
const landmarkCount = el('landmark-count');
const keyframeCount = el('keyframe-count');
const fpsLabel = el('fps');
const timingsPanel = el('timings');
const events = el('events');

/**
 * Which backend to run.
 *
 * At M0 this is the stub. At M7 it becomes:
 *
 * ```ts
 * const wasm = await import('web-slam/wasm');
 * return await wasm.WasmBackend.load(await import('../pkg/wslam_wasm.js'));
 * ```
 *
 * The rest of this file does not change, which is the property the M0 mock
 * exists to prove.
 */
async function createBackend() {
  type WasmModule = import('web-slam/wasm').WasmModule;
  // Prefer the real pipeline; fall back to the mock so the demo still runs
  // when the wasm artifact has not been built. `?backend=mock` forces the mock.
  const forced = new URLSearchParams(location.search).get('backend');
  if (forced !== 'mock') {
    try {
      const [{ WasmBackend }, wasm] = await Promise.all([
        import('web-slam/wasm'),
        import('web-slam/pkg/wslam_wasm.js'),
      ]);
      // The ambient declaration is intentionally loose; WasmBackend takes the
      // module structurally.
      const backend = await WasmBackend.load(wasm as unknown as WasmModule);
      console.info('[demo] using the wasm backend');
      // gpuActive is only meaningful after configure(); log it then.
      queueMicrotask(() =>
        console.info(
          `[demo] image front-end: ${backend.gpuActive ? 'WebGPU' : 'CPU reference'}`
        )
      );
      return backend;
    } catch (err) {
      console.warn('[demo] wasm backend unavailable, falling back to mock:', err);
    }
  }
  const { MockBackend } = await import('web-slam/mock');
  console.info('[demo] using the mock backend');
  return new MockBackend();
}

function describeState(state: TrackingState): { key: string; label: string } {
  if (isLimited(state)) return { key: 'limited', label: `limited · ${state.limited}` };
  return { key: state, label: state };
}

function announce(text: string, kind: 'accept' | 'reject' | 'plain' = 'plain'): void {
  const node = document.createElement('div');
  node.className = `event ${kind}`;
  node.textContent = text;
  events.append(node);
  // Keep at most four; the HUD is not a log viewer.
  while (events.childElementCount > 4) events.firstElementChild?.remove();
  setTimeout(() => node.remove(), 6000);
}

/** Percent-formatted 1-sigma scale uncertainty, or the honest "up to scale". */
function describeScale(pose: Pose): string {
  if (pose.scale.source === 'none') return 'SCALE NONE · UP TO SCALE';
  const pct = 100 * Math.sqrt(pose.scale.variance);
  return `SCALE ${pose.scale.source} ±${pct.toFixed(2)}%`;
}

function renderTimings(t: Record<string, number>): void {
  const stages = ['upload', 'pyramid', 'corners', 'flow', 'pnp'] as const;
  const budget = 16.6;
  timingsPanel.replaceChildren(
    ...stages.map((stage) => {
      const value = t[stage] ?? 0;
      const wrap = document.createElement('div');
      const label = document.createElement('span');
      label.textContent = stage;
      const number = document.createElement('b');
      number.textContent = `${value.toFixed(2)}ms`;
      const bar = document.createElement('div');
      bar.className = 'bar';
      bar.style.width = `${Math.min(100, (value / budget) * 100)}%`;
      wrap.append(label, number, bar);
      return wrap;
    }),
  );
}

async function main(): Promise<void> {
  const scene = new SceneView(canvas);
  const backend = await createBackend();

  const slam = await WebSlam.create({
    video,
    // The honest default. Switch to `ScaleSource.fiducial({...})` and the badge
    // starts reporting metres — nothing else in this file changes.
    scale: ScaleSource.none(),
    backend,
  });

  // Debug handles. The demo is the only integration test that exercises the
  // browser path end to end, and without a way to inspect it from a console a
  // silent failure has no diagnostic at all.
  (globalThis as Record<string, unknown>).__wslam = { slam, backend };

  const sync = new CameraSync(scene.camera, { holdOnLoss: true });

  // The AR pane is driven by the pose, so a drifting map and a perfect one
  // look identical through it. The trajectory pane beneath puts the
  // reconstruction in front of you: the trail, the landmark cloud, the
  // keyframe frusta, and the live pose. If the live frustum wanders off the
  // keyframe trail, that is the drift — permanently on screen, not behind a
  // toggle.
  //
  // Drag to orbit, pinch or wheel to zoom — gestures belong to the bottom
  // pane only. Deliberately hand-rolled: the demo ships to a phone and
  // OrbitControls is a large dependency for two gestures.
  const inBottomPane = (y: number): boolean => y > innerHeight / 2;
  let drag: { x: number; y: number } | null = null;
  let pinch = 0;
  canvas.addEventListener('pointerdown', (e) => {
    if (inBottomPane(e.clientY)) drag = { x: e.clientX, y: e.clientY };
  });
  addEventListener('pointerup', () => (drag = null));
  addEventListener('pointermove', (e) => {
    if (!drag) return;
    scene.orbitBy((e.clientX - drag.x) * -0.008, (e.clientY - drag.y) * -0.008);
    drag = { x: e.clientX, y: e.clientY };
  });
  canvas.addEventListener('wheel', (e) => {
    if (!inBottomPane(e.clientY)) return;
    e.preventDefault();
    scene.orbitBy(0, 0, e.deltaY > 0 ? 1.1 : 0.9);
  }, { passive: false });
  canvas.addEventListener('touchmove', (e) => {
    if (e.touches.length !== 2 || !inBottomPane(e.touches[0].clientY)) return;
    const [a, b] = [e.touches[0], e.touches[1]];
    const d = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
    if (pinch) scene.orbitBy(0, 0, pinch / d);
    pinch = d;
  }, { passive: true });
  addEventListener('touchend', () => (pinch = 0));

  slam.onState((state) => {
    const { key, label } = describeState(state);
    stateDot.dataset.state = key;
    stateLabel.textContent = label;
  });

  slam.onRelocalize(({ mapPoseId }) => {
    // The moment worth showing off: cover the lens, uncover it, and the session
    // recovers into the map instead of dying (spec.md §8, public demo).
    announce(`relocalized into keyframe ${mapPoseId}`, 'accept');
    scene.flashRelocalization();
  });

  slam.onLoopClosure(({ accepted, candidateId, score }) => {
    announce(
      `${accepted ? 'loop closed' : 'loop rejected'} · kf ${candidateId} · score ${score.toFixed(2)}`,
      accepted ? 'accept' : 'reject',
    );
  });

  startButton.addEventListener('click', async () => {
    startButton.disabled = true;
    gateError.textContent = '';
    try {
      // Must be inside the click handler — iOS grants motion permission only
      // from a user gesture. spec.md §3 makes this explicit rather than hiding
      // it behind a promise that mysteriously never resolves.
      await slam.start();
      gate.hidden = true;
      loop();
    } catch (err) {
      startButton.disabled = false;
      gateError.textContent =
        err instanceof Error ? err.message : 'could not start the camera';
    }
  });

  let frames = 0;
  let fpsWindowStart = performance.now();

  function loop(): void {
    requestAnimationFrame(loop);

    // ---- the entire default integration ----
    const pose = slam.currentPose();
    if (pose) sync.update(pose);
    if (pose) scene.setLivePose(pose.matrix);
    // ----------------------------------------

    if (pose) {
      scaleBadge.textContent = describeScale(pose);
      scene.setUncertainty(pose);
    }

    // Poll the state rather than relying only on `onState`. That event fires
    // on transitions, so a session that never leaves `initializing` leaves the
    // label showing the page's initial "idle" and looks like nothing is
    // running at all — which is indistinguishable from a real hang.
    const { key, label } = describeState(slam.state);
    stateDot.dataset.state = key;
    stateLabel.textContent = label;

    const debug = slam.debug;
    scene.setLandmarks(debug.landmarks());
    scene.setTrajectory(debug.trajectory());
    scene.setKeyframes(debug.keyframes());
    scene.render();

    // Features vs landmarks separates the two ways this fails on a real
    // device: no features means the image has nothing to track (too dark, too
    // smooth, out of focus); features but no landmarks means the two-view
    // bootstrap has not seen enough parallax yet, which needs sideways motion
    // rather than rotation.
    const featureCount = debug.features().length;
    landmarkCount.textContent = `${featureCount} FEAT · ${debug.landmarks().length / 3} LANDMARKS`;
    keyframeCount.textContent = `${debug.keyframes().length} KEYFRAMES`;
    // Path length in map units — honest about scale: the unit suffix only
    // claims metres when a scale source has actually anchored the session.
    const metric = pose ? pose.scale.source !== 'none' : false;
    trailLength.textContent = `PATH ${scene.trailLength().toFixed(1)} ${metric ? 'M' : 'U'}`;
    renderTimings(debug.timings() as unknown as Record<string, number>);

    frames++;
    const now = performance.now();
    if (now - fpsWindowStart > 500) {
      fpsLabel.textContent = `${Math.round((frames * 1000) / (now - fpsWindowStart))} fps`;
      frames = 0;
      fpsWindowStart = now;
    }
  }
}

main().catch((err) => {
  gateError.textContent = err instanceof Error ? err.message : String(err);
});
