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
  if (pose.scale.source === 'none') return 'scale: none · up to scale';
  const pct = 100 * Math.sqrt(pose.scale.variance);
  return `scale: ${pose.scale.source} · ±${pct.toFixed(2)}%`;
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

  const sync = new CameraSync(scene.camera, { holdOnLoss: true });

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
    // ----------------------------------------

    if (pose) {
      scaleBadge.textContent = describeScale(pose);
      scene.setUncertainty(pose);
    }

    const debug = slam.debug;
    scene.setLandmarks(debug.landmarks());
    scene.setTrajectory(debug.trajectory());
    scene.setKeyframes(debug.keyframes());
    scene.render();

    landmarkCount.textContent = `${debug.landmarks().length / 3} landmarks`;
    keyframeCount.textContent = `${debug.keyframes().length} keyframes`;
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
