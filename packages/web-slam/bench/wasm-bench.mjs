// Benchmark the real wasm artifact under Node.
//
//   node packages/web-slam/bench/wasm-bench.mjs [frames]
//
// Node stands in for the browser: the same wasm-bindgen web-target module
// loads because `init` accepts raw bytes (no fetch), and Node's global
// `performance` feeds the pipeline's stage clock. It is not a phone — treat
// the numbers as an A/B instrument for build-flag and code changes, not as
// absolute mobile truth. The scene is a seeded random-dot texture translating
// at 2 px/frame: enough structure for FAST and KLT to do full work, which is
// what the budget is (flow + corners are ~98% of the native frame).
//
// Prints per-stage means over the steady-state tail and the frame-total
// median/p99, plus a machine-readable JSON line for diffing runs.

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const WIDTH = 640;
const HEIGHT = 360;
const FRAMES = Number(process.argv[2] ?? 300);
const WARMUP = 30;

const pkg = new URL('../pkg/', import.meta.url);
const init = (await import(new URL('wslam_wasm.js', pkg).href)).default;
const { WasmSlam, init_logging } = await import(new URL('wslam_wasm.js', pkg).href);
await init(readFileSync(fileURLToPath(new URL('wslam_wasm_bg.wasm', pkg))));
init_logging?.('warn');

// Deterministic scene: band-limited random dots, translated per frame with
// wraparound. Seeded so every run and every build measures identical work.
function makeTexture(seed) {
  const tex = new Uint8Array(WIDTH * HEIGHT);
  let s = seed >>> 0;
  const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32);
  // Sparse bright dots on a mid-grey field, then a cheap 3x3 blur so corners
  // are detectable but not single-pixel aliases.
  tex.fill(96);
  for (let i = 0; i < 4000; i++) {
    const x = 1 + Math.floor(rand() * (WIDTH - 2));
    const y = 1 + Math.floor(rand() * (HEIGHT - 2));
    const v = 160 + Math.floor(rand() * 95);
    for (let dy = -1; dy <= 1; dy++)
      for (let dx = -1; dx <= 1; dx++)
        tex[(y + dy) * WIDTH + (x + dx)] = Math.max(tex[(y + dy) * WIDTH + (x + dx)], v - 40 * (Math.abs(dx) + Math.abs(dy)));
  }
  return tex;
}

const texture = makeTexture(0x5eed);
const rgba = new Uint8Array(WIDTH * HEIGHT * 4);

function renderFrame(shift) {
  for (let y = 0; y < HEIGHT; y++) {
    const row = y * WIDTH;
    for (let x = 0; x < WIDTH; x++) {
      const v = texture[row + ((x + shift) % WIDTH)];
      const o = (row + x) * 4;
      rgba[o] = rgba[o + 1] = rgba[o + 2] = v;
      rgba[o + 3] = 255;
    }
  }
  return rgba;
}

const config = {
  scale: { kind: 'none', config: {} },
  tier: 1, // vision only: the bench feeds no motion events
  map: false, // keyframe/BoW work is backend-budgeted noise for this purpose
  seed: 0x5eed,
  width: WIDTH,
  height: HEIGHT,
  motionAvailable: false,
};
const slam = new WasmSlam(JSON.stringify(config));

const totals = [];
const stages = { upload: 0, pyramid: 0, corners: 0, flow: 0, pnp: 0 };
let measured = 0;

for (let i = 0; i < FRAMES; i++) {
  const frame = renderFrame(i * 2);
  slam.push_frame(i, i / 30, performance.now(), frame, WIDTH, HEIGHT);
  slam.step_poses();
  const t = slam.debug_timings(); // [upload, pyramid, corners, flow, pnp, total]
  if (i >= WARMUP) {
    stages.upload += t[0];
    stages.pyramid += t[1];
    stages.corners += t[2];
    stages.flow += t[3];
    stages.pnp += t[4];
    totals.push(t[5]);
    measured++;
  }
}
slam.free();

totals.sort((a, b) => a - b);
const q = (p) => totals[Math.min(totals.length - 1, Math.floor(p * totals.length))];
const mean = Object.fromEntries(
  Object.entries(stages).map(([k, v]) => [k, +(v / measured).toFixed(3)]),
);
const summary = {
  frames: measured,
  size: `${WIDTH}x${HEIGHT}`,
  stage_mean_ms: mean,
  total_ms: { median: +q(0.5).toFixed(3), p90: +q(0.9).toFixed(3), p99: +q(0.99).toFixed(3) },
};

console.log(
  `wasm ${WIDTH}x${HEIGHT}, ${measured} frames: ` +
    `median ${summary.total_ms.median} ms, p99 ${summary.total_ms.p99} ms\n` +
    `  stages: pyramid ${mean.pyramid}, corners ${mean.corners}, flow ${mean.flow}, pnp ${mean.pnp}`,
);
console.log(JSON.stringify(summary));
