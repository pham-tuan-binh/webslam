/**
 * web-slam — a 6-DoF pose source for stock mobile browsers, with an explicit
 * metric anchor and calibrated uncertainty.
 *
 * ```ts
 * import { WebSlam } from 'web-slam';
 *
 * const slam = await WebSlam.create({ video: videoEl });
 * startButton.onclick = () => slam.start();
 *
 * renderLoop(() => {
 *   const pose = slam.currentPose();
 *   if (pose) camera.matrix.fromArray(pose.matrix);
 * });
 * ```
 */

import { CameraShim, MotionShim, type RawFrame, type RawMotion } from './shim.js';
import { ScaleSource } from './scale.js';
import type { Backend } from './backend.js';
import {
  hasPose,
  isLimited,
  type LoopClosureEvent,
  type Pose,
  type RelocalizeEvent,
  type SensorTier,
  type TrackingState,
  type Unsubscribe,
  type WebSlamApi,
  type WebSlamDebug,
  type WebSlamOptions,
} from './types.js';

export * from './types.js';
export { ScaleSource } from './scale.js';
export type {
  DeclaredOptions,
  FiducialFamily,
  FiducialOptions,
  LearnedOptions,
} from './scale.js';
export type { Backend, BackendOptions, StepResult } from './backend.js';
export { CameraShim, MotionShim, requestMotionPermission } from './shim.js';
export type { RawFrame, RawMotion } from './shim.js';

/** Version of the stable public API. Matches `wslam_core::PUBLIC_API_VERSION`. */
export const VERSION = '0.1.0';

/** Version of the unstable debug surface. Deliberately separate from VERSION. */
export const DEBUG_VERSION = '0.1.0-unstable';

/** Extra options for advanced construction. Not needed on the default path. */
export interface WebSlamAdvancedOptions extends WebSlamOptions {
  /**
   * Supply a backend explicitly.
   *
   * The reason this is public: the demo and the conformance tests drive the
   * real API against the mock backend, which is how spec.md §3's "freeze it at
   * M0, against a mock" is actually enforced rather than merely intended.
   *
   * When omitted, the mock backend is loaded **dynamically**, so it does not
   * enter the static graph of a consumer who supplies their own.
   */
  backend?: Backend;
}

/** Minimal typed event emitter. Kept private; no dependency worth its bytes. */
class Emitter<T> {
  private handlers = new Set<(value: T) => void>();
  add(cb: (value: T) => void): Unsubscribe {
    this.handlers.add(cb);
    return () => {
      this.handlers.delete(cb);
    };
  }
  emit(value: T): void {
    for (const cb of this.handlers) {
      try {
        cb(value);
      } catch (err) {
        // A throwing subscriber must not stop the others or kill the frame
        // loop. Report and continue.
        console.error('[web-slam] listener threw:', err);
      }
    }
  }
  clear(): void {
    this.handlers.clear();
  }
}

/**
 * The library.
 *
 * spec.md §3 design rule 1: the default path is one screen of code. Construct
 * with `create`, call `start` from a click handler, pull `currentPose` at your
 * draw rate. Everything else is progressive disclosure.
 */
export class WebSlam implements WebSlamApi {
  private readonly options: Required<
    Pick<WebSlamOptions, 'video' | 'scale' | 'tier' | 'map' | 'seed'>
  > &
    WebSlamOptions;
  private readonly backend: Backend;
  private readonly camera: CameraShim;
  private readonly motion = new MotionShim();

  private readonly poseEmitter = new Emitter<Pose>();
  private readonly stateEmitter = new Emitter<[TrackingState, TrackingState]>();
  private readonly relocEmitter = new Emitter<RelocalizeEvent>();
  private readonly loopEmitter = new Emitter<LoopClosureEvent>();

  private latest: Pose | null = null;
  private trackingState: TrackingState = 'initializing';
  private started = false;
  private disposed = false;

  private constructor(options: WebSlamAdvancedOptions, backend: Backend) {
    this.options = {
      ...options,
      scale: options.scale ?? ScaleSource.none(),
      tier: options.tier ?? 2,
      map: options.map ?? true,
      seed: options.seed ?? 0x5eed,
    };
    this.backend = backend;
    this.camera = new CameraShim(options.video);
  }

  /**
   * Create a session. Does not touch the camera or the motion sensors — that
   * happens in {@link WebSlam.start}, which must run inside a user gesture.
   *
   * @throws if `scale` is `inertial` but the requested tier cannot supply it.
   */
  static async create(options: WebSlamAdvancedOptions): Promise<WebSlam> {
    if (!options.video) {
      throw new TypeError('WebSlam.create: `video` is required');
    }
    const scale = options.scale ?? ScaleSource.none();
    const tier = options.tier ?? 2;
    if (scale.kind === 'inertial' && tier < 3) {
      // spec.md §3: `ScaleSource.inertial()` "requires tier 3; throws if L0
      // unavailable". Failing here rather than silently falling back is the
      // whole point of declaring scale explicitly.
      throw new Error(
        'WebSlam.create: ScaleSource.inertial() requires tier 3 (tight visual-inertial). ' +
          'Pass { tier: 3 }, or choose another ScaleSource — see spec §2 for the tradeoffs.',
      );
    }
    let backend = options.backend;
    if (!backend) {
      const { MockBackend } = await import('./mock.js');
      backend = new MockBackend();
    }
    return new WebSlam(options, backend);
  }

  /**
   * Begin capture and tracking.
   *
   * **Call this from a click handler.** iOS requires a user gesture before
   * granting motion-sensor access; there is no way to hide that, so the API
   * makes it explicit rather than failing mysteriously (spec.md §3).
   *
   * If motion permission is denied the session continues at sensor tier 1
   * (vision only) rather than failing — that fallback is in the tier table in
   * spec.md §4.
   */
  async start(): Promise<void> {
    if (this.disposed) throw new Error('WebSlam: session already stopped');
    if (this.started) return;
    this.started = true;

    const track = await this.camera.open(this.options.cameraConstraints);
    const settings = track.getSettings();
    const processing = this.camera.processingDimensions();

    let motionAvailable = false;
    let tier: SensorTier = this.options.tier;
    if (tier >= 2) {
      motionAvailable = await this.motion.start((m) => this.onMotion(m));
      if (!motionAvailable) {
        console.warn(
          '[web-slam] motion permission denied; falling back to sensor tier 1 (vision only)',
        );
        tier = 1;
      }
    }

    await this.backend.configure({
      scale: this.options.scale,
      tier,
      map: this.options.map,
      seed: this.options.seed,
      focalPrior: this.options.focalPrior,
      // The size frames are *processed* at, which the camera shim may have
      // scaled down from the native capture size. Intrinsics come from these
      // dimensions, so a mismatch here is a two-fold error in the focal guess
      // and principal point.
      width: processing.width || settings.width || this.options.video.videoWidth || 1280,
      height: processing.height || settings.height || this.options.video.videoHeight || 720,
      motionAvailable,
    });

    this.camera.start((frame) => this.onFrame(frame));
  }

  /** Stop capture and release everything. The session cannot be restarted. */
  async stop(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    this.started = false;
    this.camera.stop();
    this.motion.stop();
    this.backend.dispose();
    this.poseEmitter.clear();
    this.stateEmitter.clear();
    this.relocEmitter.clear();
    this.loopEmitter.clear();
  }

  private onFrame(frame: RawFrame): void {
    this.backend.pushFrame(frame);
    const result = this.backend.step();

    for (const t of result.transitions) {
      this.trackingState = t.to;
      this.stateEmitter.emit([t.to, t.from]);
    }
    for (const pose of result.poses) {
      // Only overwrite the pull-path pose when the state says it is usable.
      // A renderer polling `currentPose()` during `relocalizing` should keep
      // the last good pose rather than snap to something meaningless.
      if (hasPose(pose.state)) this.latest = pose;
      this.poseEmitter.emit(pose);
    }
    for (const e of result.relocalizations) this.relocEmitter.emit(e);
    for (const e of result.loopClosures) this.loopEmitter.emit(e);
  }

  private onMotion(motion: RawMotion): void {
    this.backend.pushMotion(motion);
  }

  /** {@inheritDoc WebSlamApi.currentPose} */
  currentPose(): Pose | null {
    return this.latest;
  }

  /** {@inheritDoc WebSlamApi.onPose} */
  onPose(cb: (pose: Pose) => void): Unsubscribe {
    return this.poseEmitter.add(cb);
  }

  /** {@inheritDoc WebSlamApi.onState} */
  onState(cb: (state: TrackingState, prev: TrackingState) => void): Unsubscribe {
    return this.stateEmitter.add(([state, prev]) => cb(state, prev));
  }

  /** {@inheritDoc WebSlamApi.onRelocalize} */
  onRelocalize(cb: (event: RelocalizeEvent) => void): Unsubscribe {
    return this.relocEmitter.add(cb);
  }

  /** {@inheritDoc WebSlamApi.onLoopClosure} */
  onLoopClosure(cb: (event: LoopClosureEvent) => void): Unsubscribe {
    return this.loopEmitter.add(cb);
  }

  /** Current tracking state. */
  get state(): TrackingState {
    return this.trackingState;
  }

  /** {@inheritDoc WebSlamApi.saveMap} */
  saveMap(): Promise<Uint8Array> {
    return this.backend.saveMap();
  }

  /** Version of the stable public API. */
  readonly version = VERSION;

  /**
   * The debug surface. **Explicitly unstable** and versioned separately
   * (spec.md §3) — pin an exact version if you build on it.
   */
  readonly debug: WebSlamDebug = {
    landmarks: () => this.backend.debug.landmarks(),
    keyframes: () => this.backend.debug.keyframes(),
    trajectory: () => this.backend.debug.trajectory(),
    features: () => this.backend.debug.features(),
    poseGraph: () => this.backend.debug.poseGraph(),
    timings: () => this.backend.debug.timings(),
    version: DEBUG_VERSION,
  };
}

export { hasPose, isLimited };
