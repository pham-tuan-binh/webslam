/**
 * The M0 stub backend.
 *
 * spec.md §3: *"The public demo is written at M0 against a stub implementation
 * that returns synthetic poses. No tracking, no map, canned data. The point is
 * to validate the API with a real consumer before any of it exists ... If the
 * demo is awkward to write against the mock, the API is wrong, and M0 is when
 * that costs nothing."*
 *
 * So this file is not a toy. It is the instrument that tests the interface, and
 * it therefore exercises the *whole* interface: state transitions, covariance
 * that actually grows, scale provenance that matches the configured source,
 * relocalization events, rejected loop closures, and a debug surface with real
 * shapes in it. A mock that only returned a matrix would validate nothing.
 *
 * Everything is deterministic from the configured seed, so a demo bug is
 * reproducible (spec.md §6).
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
  Pose,
  Quaternion,
  TrackingState,
} from './types.js';

/** Deterministic 32-bit PRNG (mulberry32). Same seed, same demo, every time. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function quaternionFromEuler(x: number, y: number, z: number): Quaternion {
  const [cx, sx] = [Math.cos(x / 2), Math.sin(x / 2)];
  const [cy, sy] = [Math.cos(y / 2), Math.sin(y / 2)];
  const [cz, sz] = [Math.cos(z / 2), Math.sin(z / 2)];
  return {
    x: sx * cy * cz - cx * sy * sz,
    y: cx * sy * cz + sx * cy * sz,
    z: cx * cy * sz - sx * sy * cz,
    w: cx * cy * cz + sx * sy * sz,
  };
}

/** Column-major 4x4 from a quaternion and a translation. */
function matrixFrom(q: Quaternion, t: { x: number; y: number; z: number }): Float32Array {
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
    t.x,           t.y,           t.z,           1,
  ]);
}

const MOCK_LANDMARK_COUNT = 900;
const MOCK_FEATURE_COUNT = 180;

/**
 * A synthetic session: initialise, track a circular walk, briefly lose
 * tracking, relocalize, propose loop closures (one of which is rejected).
 *
 * The scripted loss is deliberate. spec.md §4 L4 calls tracking loss *"not an
 * edge case on a phone; it is a routine event"* — a demo that never loses
 * tracking would let us ship an API whose recovery path nobody had used.
 */
export class MockBackend implements Backend {
  private options: BackendOptions | null = null;
  private rand: () => number = mulberry32(1);
  private frames = 0;
  private t0 = 0;
  private state: TrackingState = 'initializing';
  private pending: StepResult = emptyStep();
  private trajectoryPoints: number[] = [];
  private landmarkData = new Float32Array(0);
  private keyframeList: DebugKeyframe[] = [];
  private edges: DebugPoseGraphEdge[] = [];
  private timingsData: DebugTimings = zeroTimings();

  async configure(options: BackendOptions): Promise<void> {
    this.options = options;
    this.rand = mulberry32(options.seed);
    // A fixed landmark cloud on a rough room shell, so the demo's point cloud
    // looks like a room rather than a gaussian blob.
    const xs = new Float32Array(MOCK_LANDMARK_COUNT * 3);
    for (let i = 0; i < MOCK_LANDMARK_COUNT; i++) {
      const theta = this.rand() * Math.PI * 2;
      const r = 2.2 + this.rand() * 1.4;
      xs[i * 3 + 0] = Math.cos(theta) * r;
      xs[i * 3 + 1] = -0.9 + this.rand() * 2.4;
      xs[i * 3 + 2] = Math.sin(theta) * r;
    }
    this.landmarkData = xs;
  }

  pushFrame(frame: RawFrame): void {
    if (!this.options) return;
    if (this.frames === 0) this.t0 = frame.arrivalMs;
    this.frames++;

    const prev = this.state;
    const next = this.scriptedState(this.frames);
    if (JSON.stringify(prev) !== JSON.stringify(next)) {
      this.state = next;
      this.pending.transitions.push({ from: prev, to: next });
      if (prev === 'relocalizing' && next === 'tracking') {
        this.pending.relocalizations.push({
          atTimestamp: frame.arrivalMs,
          mapPoseId: this.keyframeList.length > 0 ? this.keyframeList[0].id : 0,
        });
      }
    }

    // Plausible per-stage timings that add up to the total, so a consumer
    // charting them sees a sane budget breakdown rather than noise.
    const jitter = () => 0.85 + this.rand() * 0.3;
    const t = {
      upload: 0.9 * jitter(),
      pyramid: 1.4 * jitter(),
      corners: 1.1 * jitter(),
      flow: 3.2 * jitter(),
      pnp: 1.8 * jitter(),
      total: 0,
    };
    t.total = t.upload + t.pyramid + t.corners + t.flow + t.pnp;
    this.timingsData = t;

    if (typeof this.state === 'string' && !['tracking'].includes(this.state) && !this.isLimited()) {
      return;
    }

    const pose = this.synthesizePose(frame.arrivalMs);
    this.pending.poses.push(pose);
    this.trajectoryPoints.push(pose.position.x, pose.position.y, pose.position.z);

    // A keyframe every 20 frames, with an odometry edge behind it.
    if (this.frames % 20 === 0) {
      const id = this.keyframeList.length;
      this.keyframeList.push({
        id,
        timestamp: pose.timestamp,
        position: pose.position,
        rotation: pose.rotation,
        matrix: pose.matrix,
      });
      if (id > 0) {
        this.edges.push({ from: id - 1, to: id, kind: 'odometry', accepted: true, score: 1 });
      }
      // Two scripted loop candidates: one accepted, one rejected by geometric
      // verification. The rejected one is the interesting case — spec.md §5
      // makes the false-positive rate first-class, and the viewer needs to draw
      // rejections to tune the threshold by eye.
      if (id === 12) this.proposeLoop(id, 0, 0.61, true);
      if (id === 16) this.proposeLoop(id, 3, 0.44, false);
    }
  }

  private proposeLoop(from: number, to: number, score: number, accepted: boolean): void {
    this.edges.push({ from, to, kind: 'loop', accepted, score });
    this.pending.loopClosures.push({ accepted, candidateId: to, score });
  }

  pushMotion(_motion: RawMotion): void {
    // The mock does not use inertial data. Accepting and discarding it is
    // deliberate: the API contract is that pushing motion is always safe,
    // including at sensor tier 1 where nothing consumes it.
  }

  step(): StepResult {
    const out = this.pending;
    this.pending = emptyStep();
    return out;
  }

  async saveMap(): Promise<Uint8Array> {
    // Header-shaped bytes so a caller who round-trips through
    // `ScaleSource.map()` gets something with the right smell, and so a demo
    // that writes it to IndexedDB is exercising a realistic size.
    const kf = this.keyframeList.length;
    const bytes = new Uint8Array(64 + kf * 128);
    const view = new DataView(bytes.buffer);
    view.setUint32(0, 0x574d4150, false); // "WMAP"
    view.setUint16(4, 1, true); // format version
    view.setUint8(6, 0); // scale kind tag: none
    view.setUint32(8, kf, true);
    return bytes;
  }

  dispose(): void {
    this.pending = emptyStep();
    this.trajectoryPoints = [];
    this.keyframeList = [];
    this.edges = [];
  }

  private isLimited(): boolean {
    return typeof this.state === 'object';
  }

  /** The scripted session. Frame counts, not wall time, so replay matches. */
  private scriptedState(frame: number): TrackingState {
    if (frame < 30) return 'initializing';
    if (frame >= 300 && frame < 330) return { limited: 'excessive-motion' };
    if (frame >= 330 && frame < 360) return 'lost';
    if (frame >= 360 && frame < 380) return 'relocalizing';
    return 'tracking';
  }

  private synthesizePose(arrivalMs: number): Pose {
    const options = this.options!;
    const s = this.frames / 60;
    // A slow circular walk with a gentle bob — recognisable in a 3D view and
    // exercising all six degrees of freedom.
    const position = {
      x: Math.sin(s * 0.6) * 1.2,
      y: 0.05 * Math.sin(s * 2.4),
      z: Math.cos(s * 0.6) * 1.2 - 1.2,
    };
    const rotation = quaternionFromEuler(0.03 * Math.sin(s * 1.7), -s * 0.6, 0.02 * Math.cos(s * 1.3));

    // Covariance grows with drift and collapses after a relocalization, which
    // is the qualitative behaviour a consumer should be able to see in the mock.
    const sinceReloc = Math.max(0, this.frames - 380);
    const drift = this.state === 'tracking' ? Math.min(1, sinceReloc / 600) : 1;
    const posVar = 1e-4 + drift * 4e-3;
    const rotVar = 1e-6 + drift * 5e-5;
    const covariance = new Float64Array(36);
    for (let i = 0; i < 3; i++) covariance[i * 6 + i] = posVar * (i === 1 ? 0.5 : 1);
    for (let i = 3; i < 6; i++) covariance[i * 6 + i] = rotVar;

    // Scale provenance mirrors the configured source honestly: `none` reports
    // infinite variance rather than pretending to a number.
    const kind = options.scale.kind;
    const variance =
      kind === 'none'
        ? Number.POSITIVE_INFINITY
        : kind === 'declared' || kind === 'fiducial'
          ? 1e-5
          : kind === 'map'
            ? 6e-5
            : kind === 'inertial'
              ? 1e-4
              : 2.5e-3;

    return {
      timestamp: arrivalMs,
      position,
      rotation,
      matrix: matrixFrom(rotation, position),
      covariance,
      scale: { source: kind, variance },
      state: this.state,
      initAgeMs: arrivalMs - this.t0,
    };
  }

  readonly debug: BackendDebug = {
    landmarks: () => this.landmarkData,
    keyframes: () => this.keyframeList,
    trajectory: () => new Float32Array(this.trajectoryPoints),
    features: (): DebugFeature[] => {
      const width = this.options?.width ?? 1280;
      const height = this.options?.height ?? 720;
      const out: DebugFeature[] = [];
      const r = mulberry32((this.options?.seed ?? 1) + this.frames);
      for (let i = 0; i < MOCK_FEATURE_COUNT; i++) {
        const u = r();
        out.push({
          id: i,
          x: r() * width,
          y: r() * height,
          state: u < 0.08 ? 'new' : u < 0.9 ? 'tracked' : u < 0.97 ? 'outlier' : 'lost',
          age: Math.floor(r() * 60),
        });
      }
      return out;
    },
    poseGraph: () => this.edges,
    timings: () => this.timingsData,
  };
}
