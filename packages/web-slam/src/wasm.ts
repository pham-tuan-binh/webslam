/**
 * The real backend: a thin adapter over the `wslam-wasm` module.
 *
 * spec.md §7 requires the wasm boundary to be "deliberately thin — no logic, so
 * the core stays natively testable". This file honours the same rule from the
 * JavaScript side: it packs and unpacks, and makes no decisions.
 *
 * ## Wire format
 *
 * Poses cross as one packed `Float64Array`, 49 doubles per pose, because a
 * per-frame JSON round-trip at 60 Hz is a measurable fraction of the frame
 * budget. Events cross as JSON, because they are rare and their shapes vary.
 *
 * | offset | count | meaning                                     |
 * |--------|-------|---------------------------------------------|
 * | 0      | 1     | timestamp, ms in the `performance.now` domain |
 * | 1      | 3     | position xyz                                |
 * | 4      | 4     | rotation xyzw                               |
 * | 8      | 36    | covariance, row-major `[translation, rotation]` |
 * | 44     | 1     | scale kind tag (see `ScaleKind::tag`)       |
 * | 45     | 1     | scale variance                              |
 * | 46     | 1     | tracking state tag                          |
 * | 47     | 1     | limited reason tag, or -1                   |
 * | 48     | 1     | init age, ms                                |
 */

import {
  emptyStep,
  zeroTimings,
  type Backend,
  type BackendDebug,
  type BackendOptions,
  type StepResult,
} from './backend.js';
import type { RawFrame, RawMotion } from './shim.js';
import type {
  DebugFeature,
  DebugKeyframe,
  DebugPoseGraphEdge,
  DebugTimings,
  FeatureState,
  LimitedReason,
  Pose,
  ScaleKind,
  TrackingState,
} from './types.js';

/** Doubles per packed pose. Must match `wslam_wasm::POSE_STRIDE`. */
export const POSE_STRIDE = 49;

/** Tag order must match `wslam_core::ScaleKind::tag`. */
const SCALE_KINDS: ScaleKind[] = [
  'none',
  'declared',
  'fiducial',
  'learned',
  'map',
  'inertial',
];

/** Tag order must match `wslam_wasm`'s state encoding. */
const STATES = ['initializing', 'tracking', 'limited', 'relocalizing', 'lost'] as const;

const LIMITED_REASONS: LimitedReason[] = [
  'excessive-motion',
  'insufficient-features',
  'low-light',
];

const FEATURE_STATES: FeatureState[] = ['new', 'tracked', 'outlier', 'lost'];

/** The surface `wslam-wasm` exposes. Structural, so the import stays optional. */
export interface WasmModule {
  default?: (input?: unknown) => Promise<unknown>;
  WasmSlam: {
    new (configJson: string): WasmSlamHandle;
  };
}

/** Instance methods of the wasm session object. */
export interface WasmSlamHandle {
  /**
   * Claim a device previously acquired by the module's `acquire_gpu`.
   *
   * Present only when the wasm artifact was built with the `gpu` feature.
   * Returns whether one was waiting. Acquisition is a separate free function
   * because wasm-bindgen cannot bind an `async fn` taking `&mut self`.
   */
  use_gpu?(): boolean;
  push_frame(
    index: number,
    mediaTime: number,
    arrivalMs: number,
    rgba: Uint8Array,
    width: number,
    height: number,
  ): void;
  push_motion(
    index: number,
    arrivalMs: number,
    gyro: Float64Array,
    accel: Float64Array,
    intervalS: number,
  ): void;
  step_poses(): Float64Array;
  take_events(): string;
  save_map(): Uint8Array;
  debug_landmarks(): Float32Array;
  debug_trajectory(): Float32Array;
  debug_features(): Float32Array;
  debug_keyframes(): Float64Array;
  debug_pose_graph(): string;
  debug_timings(): Float64Array;
  free(): void;
}

function decodeState(stateTag: number, limitedTag: number): TrackingState {
  const name = STATES[stateTag] ?? 'lost';
  if (name === 'limited') {
    return { limited: LIMITED_REASONS[limitedTag] ?? 'insufficient-features' };
  }
  return name as TrackingState;
}

/** Column-major 4x4 from position + quaternion. Mirrors `Se3::to_matrix_f32`. */
function composeMatrix(
  p: { x: number; y: number; z: number },
  q: { x: number; y: number; z: number; w: number },
): Float32Array {
  const { x, y, z, w } = q;
  const [x2, y2, z2] = [x + x, y + y, z + z];
  const [xx, xy, xz] = [x * x2, x * y2, x * z2];
  const [yy, yz, zz] = [y * y2, y * z2, z * z2];
  const [wx, wy, wz] = [w * x2, w * y2, w * z2];
  // prettier-ignore
  return new Float32Array([
    1 - (yy + zz), xy + wz,       xz - wy,       0,
    xy - wz,       1 - (xx + zz), yz + wx,       0,
    xz + wy,       yz - wx,       1 - (xx + yy), 0,
    p.x,           p.y,           p.z,           1,
  ]);
}

/** Unpack the wire format into public {@link Pose} objects. */
export function unpackPoses(packed: Float64Array): Pose[] {
  const out: Pose[] = [];
  for (let base = 0; base + POSE_STRIDE <= packed.length; base += POSE_STRIDE) {
    const position = {
      x: packed[base + 1],
      y: packed[base + 2],
      z: packed[base + 3],
    };
    const rotation = {
      x: packed[base + 4],
      y: packed[base + 5],
      z: packed[base + 6],
      w: packed[base + 7],
    };
    out.push({
      timestamp: packed[base],
      position,
      rotation,
      matrix: composeMatrix(position, rotation),
      // Copy rather than subarray: the wasm heap can be reallocated by the next
      // allocation, which would silently detach a view we handed to a consumer.
      covariance: packed.slice(base + 8, base + 44),
      scale: {
        source: SCALE_KINDS[packed[base + 44]] ?? 'none',
        variance: packed[base + 45],
      },
      state: decodeState(packed[base + 46], packed[base + 47]),
      initAgeMs: packed[base + 48],
    });
  }
  return out;
}

/** Backend backed by the compiled Rust pipeline. */
export class WasmBackend implements Backend {
  private handle: WasmSlamHandle | null = null;
  private readonly module: WasmModule;
  private previousState: TrackingState = 'initializing';
  /** Whether the WebGPU front-end is in use. False means the CPU reference. */
  gpuActive = false;
  private gyroScratch = new Float64Array(3);
  private accelScratch = new Float64Array(3);

  constructor(module: WasmModule) {
    this.module = module;
  }

  /**
   * Load the wasm module and build a backend.
   *
   * @param init - the module's default export (its `init` function), already
   * imported by the caller. Kept as a parameter rather than a static import so
   * that bundlers can decide how to fetch the `.wasm` asset.
   */
  static async load(module: WasmModule, wasmUrl?: string): Promise<WasmBackend> {
    if (typeof module.default === 'function') {
      await module.default(wasmUrl);
    }
    return new WasmBackend(module);
  }

  async configure(options: BackendOptions): Promise<void> {
    // Uint8Array is not JSON-serialisable; the map bytes travel separately.
    const { bytes, ...scaleConfig } = options.scale.config as {
      bytes?: Uint8Array;
    } & Record<string, unknown>;
    const config = {
      scale: { kind: options.scale.kind, config: scaleConfig },
      tier: options.tier,
      map: options.map,
      seed: options.seed,
      focalPrior: options.focalPrior ?? null,
      width: options.width,
      height: options.height,
      motionAvailable: options.motionAvailable,
      mapBytesLength: bytes?.byteLength ?? 0,
    };
    this.handle = new this.module.WasmSlam(JSON.stringify(config));
    // Must happen before the first frame: the tracker builds its GPU pipeline
    // lazily from whatever context it holds when a frame first arrives.
    const acquire = (this.module as { acquire_gpu?: () => Promise<boolean> })
      .acquire_gpu;
    if (typeof acquire === 'function' && typeof this.handle.use_gpu === 'function') {
      this.gpuActive = (await acquire()) && this.handle.use_gpu();
    }
  }

  pushFrame(frame: RawFrame): void {
    // `Uint8ClampedArray` and `Uint8Array` share a buffer layout, so this is a
    // view, not a copy — the copy happens once inside wasm-bindgen.
    this.handle?.push_frame(
      frame.index,
      frame.mediaTime,
      frame.arrivalMs,
      new Uint8Array(frame.rgba.buffer, frame.rgba.byteOffset, frame.rgba.byteLength),
      frame.width,
      frame.height,
    );
  }

  pushMotion(motion: RawMotion): void {
    if (!this.handle) return;
    this.gyroScratch.set(motion.gyroDegPerS);
    this.accelScratch.set(motion.accelWithGravity);
    this.handle.push_motion(
      motion.index,
      motion.arrivalMs,
      this.gyroScratch,
      this.accelScratch,
      motion.intervalS,
    );
  }

  step(): StepResult {
    if (!this.handle) return emptyStep();
    const result = emptyStep();
    result.poses = unpackPoses(this.handle.step_poses());

    for (const pose of result.poses) {
      if (JSON.stringify(pose.state) !== JSON.stringify(this.previousState)) {
        result.transitions.push({ from: this.previousState, to: pose.state });
        this.previousState = pose.state;
      }
    }

    const raw = this.handle.take_events();
    if (raw && raw !== '[]') {
      for (const event of JSON.parse(raw) as WireEvent[]) {
        if (event.type === 'relocalized') {
          result.relocalizations.push({
            atTimestamp: event.atTimestamp,
            mapPoseId: event.keyframe,
          });
        } else if (event.type === 'loop') {
          result.loopClosures.push({
            accepted: event.accepted,
            candidateId: event.candidate,
            score: event.score,
          });
        }
      }
    }
    return result;
  }

  async saveMap(): Promise<Uint8Array> {
    if (!this.handle) throw new Error('web-slam: session not started');
    return this.handle.save_map();
  }

  dispose(): void {
    this.handle?.free();
    this.handle = null;
  }

  readonly debug: BackendDebug = {
    landmarks: () => this.handle?.debug_landmarks() ?? new Float32Array(0),
    trajectory: () => this.handle?.debug_trajectory() ?? new Float32Array(0),
    features: (): DebugFeature[] => {
      // Packed: [id, x, y, stateTag, age] per feature.
      const packed = this.handle?.debug_features() ?? new Float32Array(0);
      const out: DebugFeature[] = [];
      for (let i = 0; i + 5 <= packed.length; i += 5) {
        out.push({
          id: packed[i],
          x: packed[i + 1],
          y: packed[i + 2],
          state: FEATURE_STATES[packed[i + 3]] ?? 'lost',
          age: packed[i + 4],
        });
      }
      return out;
    },
    keyframes: (): DebugKeyframe[] => {
      // Packed: [id, timestamp, px, py, pz, qx, qy, qz, qw] per keyframe.
      const packed = this.handle?.debug_keyframes() ?? new Float64Array(0);
      const out: DebugKeyframe[] = [];
      for (let i = 0; i + 9 <= packed.length; i += 9) {
        const position = { x: packed[i + 2], y: packed[i + 3], z: packed[i + 4] };
        const rotation = {
          x: packed[i + 5],
          y: packed[i + 6],
          z: packed[i + 7],
          w: packed[i + 8],
        };
        out.push({
          id: packed[i],
          timestamp: packed[i + 1],
          position,
          rotation,
          matrix: composeMatrix(position, rotation),
        });
      }
      return out;
    },
    poseGraph: (): DebugPoseGraphEdge[] => {
      const raw = this.handle?.debug_pose_graph();
      return raw ? (JSON.parse(raw) as DebugPoseGraphEdge[]) : [];
    },
    timings: (): DebugTimings => {
      const t = this.handle?.debug_timings();
      if (!t || t.length < 6) return zeroTimings();
      return {
        upload: t[0],
        pyramid: t[1],
        corners: t[2],
        flow: t[3],
        pnp: t[4],
        total: t[5],
      };
    },
  };
}

type WireEvent =
  | { type: 'relocalized'; atTimestamp: number; keyframe: number }
  | { type: 'loop'; accepted: boolean; candidate: number; score: number };
