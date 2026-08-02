/**
 * API conformance tests, driven through the real `WebSlam` object against
 * {@link MockBackend}.
 *
 * spec.md §3: *"If the demo is awkward to write against the mock, the API is
 * wrong, and M0 is when that costs nothing."* These tests are the automated
 * half of that check — they assert the properties the spec promises about the
 * interface, not the correctness of any estimator.
 */

import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';
import { MockBackend } from '../src/mock.js';
import { ScaleSource } from '../src/scale.js';
import { WebSlam, VERSION, DEBUG_VERSION } from '../src/index.js';
import { hasPose, isLimited, type Pose, type TrackingState } from '../src/types.js';
import { unpackPoses, POSE_STRIDE } from '../src/wasm.js';
import { positionUncertaintyEllipsoid, CameraSync } from '../src/three.js';

/**
 * Drive a backend directly with synthetic frames.
 *
 * `WebSlam.start()` needs `getUserMedia`, which does not exist under vitest, so
 * the interface tests drive the backend seam and a small number of separate
 * tests cover the browser plumbing. That split mirrors spec.md §6 Tier 4:
 * "keep them separable ... so a failure identifies which of the two it is."
 */
async function runFrames(backend: MockBackend, count: number, options = {}) {
  await backend.configure({
    scale: ScaleSource.none(),
    tier: 2,
    map: true,
    seed: 42,
    width: 1280,
    height: 720,
    motionAvailable: true,
    ...options,
  });
  const collected: Pose[] = [];
  const transitions: { from: TrackingState; to: TrackingState }[] = [];
  const relocs: number[] = [];
  const loops: { accepted: boolean; score: number }[] = [];
  for (let i = 0; i < count; i++) {
    backend.pushFrame({
      index: i,
      mediaTime: i / 60,
      arrivalMs: (i * 1000) / 60,
      rgba: new Uint8ClampedArray(4),
      width: 1280,
      height: 720,
    });
    const step = backend.step();
    collected.push(...step.poses);
    transitions.push(...step.transitions);
    relocs.push(...step.relocalizations.map((r) => r.atTimestamp));
    loops.push(...step.loopClosures);
  }
  return { collected, transitions, relocs, loops };
}

describe('the 90% case', () => {
  it('constructs without touching the camera', async () => {
    const video = {} as HTMLVideoElement;
    const slam = await WebSlam.create({ video, backend: new MockBackend() });
    expect(slam.currentPose()).toBeNull();
    expect(slam.state).toBe('initializing');
    expect(slam.version).toBe(VERSION);
  });

  it('rejects construction without a video element', async () => {
    // @ts-expect-error deliberately omitting the required option
    await expect(WebSlam.create({})).rejects.toThrow(/video/);
  });

  it('produces a renderer-ready column-major matrix', async () => {
    const backend = new MockBackend();
    const { collected } = await runFrames(backend, 60);
    const pose = collected.at(-1)!;
    expect(pose.matrix).toBeInstanceOf(Float32Array);
    expect(pose.matrix.length).toBe(16);
    // Translation lives at indices 12..14 in column-major order. If this ever
    // reads as 3..5, every consumer's camera is transposed.
    expect(pose.matrix[12]).toBeCloseTo(pose.position.x, 5);
    expect(pose.matrix[13]).toBeCloseTo(pose.position.y, 5);
    expect(pose.matrix[14]).toBeCloseTo(pose.position.z, 5);
    expect(pose.matrix[15]).toBe(1);
  });
});

describe('covariance and scale provenance travel with every pose', () => {
  it('never emits a pose without both', async () => {
    const { collected } = await runFrames(new MockBackend(), 400);
    expect(collected.length).toBeGreaterThan(100);
    for (const pose of collected) {
      expect(pose.covariance).toBeInstanceOf(Float64Array);
      expect(pose.covariance.length).toBe(36);
      expect(pose.scale.source).toBeDefined();
      expect(pose.scale.variance).toBeDefined();
    }
  });

  it('reports infinite variance for the up-to-scale default', async () => {
    const { collected } = await runFrames(new MockBackend(), 60, {
      scale: ScaleSource.none(),
    });
    for (const pose of collected) {
      expect(pose.scale.source).toBe('none');
      // The design invariant: `none` must not masquerade as a confident 1.0.
      expect(pose.scale.variance).toBe(Number.POSITIVE_INFINITY);
    }
  });

  it('reports the configured source, not a guess', async () => {
    for (const spec of [
      ScaleSource.declared({ distanceMeters: 1 }),
      ScaleSource.fiducial({ family: 'apriltag36h11', sizeMeters: 0.1 }),
      ScaleSource.map(new Uint8Array([1, 2, 3])),
    ]) {
      const { collected } = await runFrames(new MockBackend(), 60, { scale: spec });
      expect(collected.at(-1)!.scale.source).toBe(spec.kind);
      expect(collected.at(-1)!.scale.variance).toBeLessThan(Infinity);
    }
  });

  it('gives a map-derived scale more variance than an exact ruler', async () => {
    const exact = await runFrames(new MockBackend(), 60, {
      scale: ScaleSource.fiducial({ family: 'apriltag36h11', sizeMeters: 0.1 }),
    });
    const inherited = await runFrames(new MockBackend(), 60, {
      scale: ScaleSource.map(new Uint8Array([1])),
    });
    // spec.md §4 L5: map "must not report itself as more certain than its origin".
    expect(inherited.collected.at(-1)!.scale.variance).toBeGreaterThan(
      exact.collected.at(-1)!.scale.variance,
    );
  });

  it('produces a symmetric covariance with a non-negative diagonal', async () => {
    const { collected } = await runFrames(new MockBackend(), 120);
    for (const pose of collected) {
      for (let i = 0; i < 6; i++) {
        expect(pose.covariance[i * 6 + i]).toBeGreaterThanOrEqual(0);
        for (let j = 0; j < 6; j++) {
          expect(pose.covariance[i * 6 + j]).toBeCloseTo(pose.covariance[j * 6 + i], 12);
        }
      }
    }
  });
});

describe('the state machine is exercised, not decorative', () => {
  it('runs the full initialise -> track -> lose -> relocalize cycle', async () => {
    const { transitions, relocs } = await runFrames(new MockBackend(), 420);
    const names = transitions.map((t) => (isLimited(t.to) ? `limited:${t.to.limited}` : t.to));
    expect(names).toContain('tracking');
    expect(names).toContain('limited:excessive-motion');
    expect(names).toContain('lost');
    expect(names).toContain('relocalizing');
    // Recovery must actually announce itself; a silent recovery is a bug.
    expect(relocs.length).toBeGreaterThan(0);
  });

  it('withholds pose only in the states that promise none', async () => {
    const { collected } = await runFrames(new MockBackend(), 420);
    for (const pose of collected) {
      expect(hasPose(pose.state)).toBe(true);
    }
  });
});

describe('loop closures report rejections too', () => {
  it('surfaces both accepted and rejected candidates', async () => {
    const { loops } = await runFrames(new MockBackend(), 400);
    expect(loops.some((l) => l.accepted)).toBe(true);
    // spec.md §5 makes the false-positive rate first-class. A consumer that
    // cannot see rejected candidates cannot tune the threshold.
    expect(loops.some((l) => !l.accepted)).toBe(true);
  });

  it('exposes rejected edges in the pose graph', async () => {
    const backend = new MockBackend();
    await runFrames(backend, 400);
    const edges = backend.debug.poseGraph();
    expect(edges.some((e) => e.kind === 'loop' && !e.accepted)).toBe(true);
    expect(edges.some((e) => e.kind === 'odometry')).toBe(true);
  });
});

describe('pull and push are genuinely different paths', () => {
  it('push delivers every tracked frame; pull delivers only the freshest', async () => {
    const backend = new MockBackend();
    const slam = await WebSlam.create({ video: {} as HTMLVideoElement, backend });
    const pushed: Pose[] = [];
    slam.onPose((p) => pushed.push(p));

    await backend.configure({
      scale: ScaleSource.none(),
      tier: 2,
      map: true,
      seed: 1,
      width: 640,
      height: 480,
      motionAvailable: true,
    });

    // `start()` needs getUserMedia, which vitest has no business providing.
    // Drive the frame handler the way the shim would instead — that is the
    // seam under test.
    const deliver = (slam as unknown as { onFrame(f: unknown): void }).onFrame.bind(slam);
    for (let i = 0; i < 200; i++) {
      backend.pushFrame({
        index: i,
        mediaTime: i / 60,
        arrivalMs: i * 16.6,
        rgba: new Uint8ClampedArray(4),
        width: 640,
        height: 480,
      });
      deliver({ index: i, mediaTime: i / 60, arrivalMs: i * 16.6, rgba: new Uint8ClampedArray(4), width: 640, height: 480 });
    }

    // Push: a recorder must see every tracked frame.
    expect(pushed.length).toBeGreaterThan(100);
    // Pull: the renderer sees one pose, and it is the newest.
    expect(slam.currentPose()).not.toBeNull();
    expect(slam.currentPose()!.timestamp).toBe(pushed.at(-1)!.timestamp);
  });

  it('returns a working unsubscribe handle', async () => {
    const slam = await WebSlam.create({
      video: {} as HTMLVideoElement,
      backend: new MockBackend(),
    });
    let calls = 0;
    const unsubscribe = slam.onState(() => calls++);
    const emit = (slam as unknown as {
      stateEmitter: { emit(v: [TrackingState, TrackingState]): void };
    }).stateEmitter;
    emit.emit(['tracking', 'initializing']);
    unsubscribe();
    emit.emit(['lost', 'tracking']);
    expect(calls).toBe(1);
  });

  it('an exception in one listener does not stop the others', async () => {
    const slam = await WebSlam.create({
      video: {} as HTMLVideoElement,
      backend: new MockBackend(),
    });
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const seen: string[] = [];
    slam.onState(() => {
      throw new Error('boom');
    });
    slam.onState(() => seen.push('second'));
    (slam as unknown as {
      stateEmitter: { emit(v: [TrackingState, TrackingState]): void };
    }).stateEmitter.emit(['tracking', 'initializing']);
    expect(seen).toEqual(['second']);
    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });
});

describe('ScaleSource validates its arguments', () => {
  it('rejects a non-positive fiducial size', () => {
    expect(() => ScaleSource.fiducial({ family: 'apriltag36h11', sizeMeters: 0 })).toThrow(
      RangeError,
    );
  });

  it('rejects an empty saved map', () => {
    expect(() => ScaleSource.map(new Uint8Array(0))).toThrow(TypeError);
  });

  it('rejects a negative declared distance', () => {
    expect(() => ScaleSource.declared({ distanceMeters: -1 })).toThrow(RangeError);
  });

  it('refuses inertial scale below tier 3 instead of degrading silently', async () => {
    await expect(
      WebSlam.create({
        video: {} as HTMLVideoElement,
        scale: ScaleSource.inertial(),
        tier: 2,
      }),
    ).rejects.toThrow(/tier 3/);
  });

  it('accepts inertial scale at tier 3', async () => {
    const slam = await WebSlam.create({
      video: {} as HTMLVideoElement,
      scale: ScaleSource.inertial(),
      tier: 3,
      backend: new MockBackend(),
    });
    expect(slam).toBeDefined();
  });

  it('freezes its config so a caller cannot mutate it after the fact', () => {
    const spec = ScaleSource.fiducial({ family: 'apriltag36h11', sizeMeters: 0.1 });
    expect(() => {
      (spec.config as Record<string, unknown>).sizeMeters = 99;
    }).toThrow();
  });
});

describe('the debug surface', () => {
  it('is versioned separately from the public API', async () => {
    const slam = await WebSlam.create({
      video: {} as HTMLVideoElement,
      backend: new MockBackend(),
    });
    expect(slam.debug.version).toBe(DEBUG_VERSION);
    expect(slam.debug.version).not.toBe(slam.version);
  });

  it('returns the shapes the viewer needs', async () => {
    const backend = new MockBackend();
    await runFrames(backend, 200);
    expect(backend.debug.landmarks().length % 3).toBe(0);
    expect(backend.debug.landmarks().length).toBeGreaterThan(0);
    expect(backend.debug.trajectory().length % 3).toBe(0);
    expect(backend.debug.keyframes().length).toBeGreaterThan(0);
    const features = backend.debug.features();
    expect(features.length).toBeGreaterThan(0);
    // Per-feature state exists so the overlay can colour by it.
    expect(new Set(features.map((f) => f.state)).size).toBeGreaterThan(1);
    const t = backend.debug.timings();
    expect(t.total).toBeCloseTo(t.upload + t.pyramid + t.corners + t.flow + t.pnp, 6);
  });
});

describe('the wasm wire format', () => {
  it('round-trips a pose through the packed layout', () => {
    const packed = new Float64Array(POSE_STRIDE);
    packed[0] = 1234.5;
    packed.set([1, 2, 3], 1);
    packed.set([0, 0, 0, 1], 4);
    for (let i = 0; i < 6; i++) packed[8 + i * 6 + i] = 0.25;
    packed[44] = 2; // fiducial
    packed[45] = 1e-5;
    packed[46] = 1; // tracking
    packed[47] = -1;
    packed[48] = 900;

    const [pose] = unpackPoses(packed);
    expect(pose.timestamp).toBe(1234.5);
    expect(pose.position).toEqual({ x: 1, y: 2, z: 3 });
    expect(pose.scale.source).toBe('fiducial');
    expect(pose.state).toBe('tracking');
    expect(pose.covariance[0]).toBe(0.25);
    expect(pose.matrix[12]).toBe(1);
  });

  it('decodes the limited state with its reason', () => {
    const packed = new Float64Array(POSE_STRIDE);
    packed.set([0, 0, 0, 1], 4);
    packed[46] = 2; // limited
    packed[47] = 2; // low-light
    const [pose] = unpackPoses(packed);
    expect(pose.state).toEqual({ limited: 'low-light' });
  });

  it('copies covariance out of the wasm heap rather than viewing it', () => {
    const packed = new Float64Array(POSE_STRIDE);
    packed.set([0, 0, 0, 1], 4);
    packed[8] = 5;
    const [pose] = unpackPoses(packed);
    packed[8] = 999;
    // A subarray view would have followed the mutation, and would detach
    // entirely if the wasm heap grew.
    expect(pose.covariance[0]).toBe(5);
  });

  it('ignores a trailing partial record rather than emitting garbage', () => {
    const packed = new Float64Array(POSE_STRIDE + 7);
    packed.set([0, 0, 0, 1], 4);
    expect(unpackPoses(packed)).toHaveLength(1);
  });
});

describe('the three.js helper', () => {
  let camera: {
    matrix: { fromArray: (a: ArrayLike<number>) => unknown; last?: ArrayLike<number> };
    matrixAutoUpdate: boolean;
    fov?: number;
    aspect?: number;
    updateProjectionMatrix?: () => void;
  };

  beforeEach(() => {
    camera = {
      matrix: {
        fromArray(a: ArrayLike<number>) {
          camera.matrix.last = Array.from(a);
          return this;
        },
      },
      matrixAutoUpdate: true,
      updateProjectionMatrix: vi.fn(),
    };
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('disables matrixAutoUpdate so three does not recompose over our matrix', () => {
    new CameraSync(camera);
    expect(camera.matrixAutoUpdate).toBe(false);
  });

  it('holds the last good pose through a loss by default', async () => {
    const sync = new CameraSync(camera);
    const { collected } = await runFrames(new MockBackend(), 60);
    sync.update(collected.at(-1)!);
    const held = camera.matrix.last;
    sync.update({ ...collected.at(-1)!, state: 'lost' });
    expect(camera.matrix.last).toEqual(held);
  });

  it('tolerates a null pose', () => {
    const sync = new CameraSync(camera);
    expect(() => sync.update(null)).not.toThrow();
  });

  it('rejects an out-of-range smoothing factor', () => {
    expect(() => new CameraSync(camera, { smoothing: 0 })).toThrow(RangeError);
    expect(() => new CameraSync(camera, { smoothing: 1.5 })).toThrow(RangeError);
  });

  it('derives vertical FOV from the estimated focal length', () => {
    const sync = new CameraSync(camera);
    // f = h/2 / tan(vfov/2); a 45-degree vertical FOV over 720 px means
    // f = 360 / tan(22.5deg) = 869.0.
    sync.setIntrinsics(869.0, 1280, 720);
    expect(camera.fov).toBeCloseTo(45, 1);
    expect(camera.aspect).toBeCloseTo(1280 / 720, 6);
    expect(camera.updateProjectionMatrix).toHaveBeenCalled();
  });

  it('recovers the principal axes of an uncertainty ellipsoid', () => {
    // Diagonal covariance: axes are the square roots, in some order.
    const cov = new Float64Array(36);
    cov[0] = 0.04; // sigma_x = 0.2
    cov[7] = 0.01; // sigma_y = 0.1
    cov[14] = 0.09; // sigma_z = 0.3
    const { axes, basis } = positionUncertaintyEllipsoid(cov);
    expect([...axes].sort((a, b) => a - b).map((v) => +v.toFixed(6))).toEqual([0.1, 0.2, 0.3]);
    expect(basis.length).toBe(9);
  });

  it('scales the ellipsoid by the requested sigma', () => {
    const cov = new Float64Array(36);
    cov[0] = cov[7] = cov[14] = 0.25; // sigma = 0.5 isotropic
    const { axes } = positionUncertaintyEllipsoid(cov, 2);
    for (const a of axes) expect(a).toBeCloseTo(1.0, 9);
  });
});
