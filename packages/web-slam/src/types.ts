/**
 * The public type surface. spec.md §3: "This is the product. Everything below
 * it is implementation."
 *
 * Nothing in this file may reference the wasm module, the backend, or any
 * browser API. It is the contract, and the demo was written against it at M0
 * before an implementation existed.
 */

/** A 3-vector. Plain object, so it interops with every renderer without a cast. */
export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

/** A quaternion, `w` last to match three.js and WebXR. */
export interface Quaternion {
  x: number;
  y: number;
  z: number;
  w: number;
}

/** Which ruler produced metric scale. See spec.md §2 for the tradeoffs of each. */
export type ScaleKind =
  | 'none'
  | 'declared'
  | 'fiducial'
  | 'learned'
  | 'map'
  | 'inertial';

/** Why tracking is degraded but still producing pose. */
export type LimitedReason =
  | 'excessive-motion'
  | 'insufficient-features'
  | 'low-light';

/**
 * Tracking state, exactly as specified in spec.md §3.
 *
 * `limited` is an object rather than a string so the reason cannot be dropped
 * by a consumer who only pattern-matches the common cases.
 */
export type TrackingState =
  | 'initializing'
  | 'tracking'
  | { limited: LimitedReason }
  | 'relocalizing'
  | 'lost';

/** Narrowing helper for the object arm of {@link TrackingState}. */
export function isLimited(
  state: TrackingState,
): state is { limited: LimitedReason } {
  return typeof state === 'object' && state !== null && 'limited' in state;
}

/** Whether a pose emitted in this state is safe to render with. */
export function hasPose(state: TrackingState): boolean {
  return state === 'tracking' || isLimited(state);
}

/**
 * A 6-DoF pose with everything needed to decide how much to trust it.
 *
 * spec.md §3: "Covariance and scale provenance travel *with* every pose. They
 * are not queried separately, because separate queries get skipped."
 */
export interface Pose {
  /** Capture time, mapped into the `performance.now()` domain. */
  timestamp: number;
  /** Camera position. **Metres only when `scale.source !== 'none'`.** */
  position: Vec3;
  /** Camera orientation. Always meaningful — rotation is scale-invariant. */
  rotation: Quaternion;
  /** 4x4 column-major, renderer-ready. Assign straight to `camera.matrix`. */
  matrix: Float32Array;
  /** 6x6 covariance, `[translation, rotation]` block order, row-major. */
  covariance: Float64Array;
  /** Provenance and uncertainty of metric scale. */
  scale: {
    /** Which ruler. `'none'` means the position is up to scale. */
    source: ScaleKind;
    /** Variance of the scale multiplier. `Infinity` when `source === 'none'`. */
    variance: number;
  };
  /** Tracking state at the moment this pose was produced. */
  state: TrackingState;
  /** Milliseconds since the current tracking session initialised. */
  initAgeMs: number;
}

/** Payload of {@link WebSlamApi.onRelocalize}. */
export interface RelocalizeEvent {
  /** When the recovery happened, in the `performance.now()` domain. */
  atTimestamp: number;
  /** Which stored keyframe the session recovered into. */
  mapPoseId: number;
}

/** Payload of {@link WebSlamApi.onLoopClosure}. */
export interface LoopClosureEvent {
  /**
   * Whether geometric verification accepted the candidate.
   *
   * Rejected candidates are reported too, deliberately: spec.md §5 makes the
   * false-positive rate a first-class metric, and you cannot tune a threshold
   * you cannot see.
   */
  accepted: boolean;
  /** Keyframe id the place-recognition query proposed. */
  candidateId: number;
  /** Bag-of-words similarity score, in `[0, 1]`. */
  score: number;
}

/** Sensor tier, declared configuration rather than an assumption (spec.md §4). */
export type SensorTier =
  /** Vision only. The fallback when motion permission is denied. */
  | 1
  /** Vision + loose orientation. The baseline, and what we ship. */
  | 2
  /** Tight visual-inertial. Optional; research track R1. */
  | 3;

/** Unsubscribe handle returned by every `on*` registration. */
export type Unsubscribe = () => void;

/** A scale source, chosen by the caller. Construct via the `ScaleSource` factory. */
export interface ScaleSourceSpec {
  readonly kind: ScaleKind;
  /** Opaque configuration forwarded to the backend. */
  readonly config: Readonly<Record<string, unknown>>;
}

/** Options for {@link WebSlamApi} construction. */
export interface WebSlamOptions {
  /** The video element the camera stream is attached to. */
  video: HTMLVideoElement;
  /**
   * Scale source. Defaults to `ScaleSource.none()` — up to scale, and honest
   * about it (spec.md §1: "The library never silently guesses scale").
   */
  scale?: ScaleSourceSpec;
  /** Sensor tier. Defaults to 2. */
  tier?: SensorTier;
  /** Whether to build a keyframe map and relocalize. Defaults to `true`. */
  map?: boolean;
  /**
   * Seed for every RNG in the pipeline, including RANSAC. Fixed by default so
   * that two runs over the same input agree (spec.md §6).
   */
  seed?: number;
  /** Requested camera constraints, merged over the defaults. */
  cameraConstraints?: MediaTrackConstraints;
  /**
   * Focal length prior in pixels. Skips part of the L2 init pan when the caller
   * already knows the camera — e.g. from a previous session on the same device.
   */
  focalPrior?: number;
}

/** Per-feature state, for overlay colouring in the debug surface. */
export type FeatureState = 'new' | 'tracked' | 'outlier' | 'lost';

/** One tracked 2D feature. */
export interface DebugFeature {
  id: number;
  x: number;
  y: number;
  state: FeatureState;
  /** Frames this feature has survived. */
  age: number;
}

/** One keyframe in the map. */
export interface DebugKeyframe {
  id: number;
  timestamp: number;
  position: Vec3;
  rotation: Quaternion;
  matrix: Float32Array;
}

/** One pose-graph edge. */
export interface DebugPoseGraphEdge {
  from: number;
  to: number;
  /** `'odometry'` for sequential edges, `'loop'` for closures. */
  kind: 'odometry' | 'loop';
  /**
   * `false` for loop candidates that geometric verification rejected. They are
   * exposed so the false-positive threshold can be tuned by eye rather than by
   * guesswork (spec.md §8, dev viewer, from M5).
   */
  accepted: boolean;
  score: number;
}

/** Per-stage frame timing, in milliseconds. Required to manage the GPU budget. */
export interface DebugTimings {
  upload: number;
  pyramid: number;
  corners: number;
  flow: number;
  pnp: number;
  total: number;
}

/**
 * The debug surface. spec.md §3: namespaced, tree-shakeable, and **explicitly
 * unstable** — versioned separately from the core API so the viewer can move
 * fast without pinning us.
 *
 * @remarks Not covered by semver. Pin an exact version if you depend on it.
 */
export interface WebSlamDebug {
  /** Sparse landmark positions, packed xyz. */
  landmarks(): Float32Array;
  /** Keyframe poses and ids. */
  keyframes(): DebugKeyframe[];
  /** Estimated trajectory so far, packed xyz. */
  trajectory(): Float32Array;
  /** Current 2D features with per-feature state for overlay colouring. */
  features(): DebugFeature[];
  /** Pose-graph edges, including loop candidates that were rejected. */
  poseGraph(): DebugPoseGraphEdge[];
  /** Per-stage timings for the most recent tracked frame. */
  timings(): DebugTimings;
  /** Version of this surface. Distinct from the core API version. */
  readonly version: string;
}

/**
 * The public API. spec.md §3, design rule 1: "The default path is one screen of
 * code." Everything past `create`, `start` and `currentPose` is progressive
 * disclosure.
 */
export interface WebSlamApi {
  /**
   * Begin capture and tracking.
   *
   * iOS requires a user gesture before motion sensors, so this is explicit and
   * **must be called from a click handler**. There is no way to hide it
   * (spec.md §3).
   */
  start(): Promise<void>;

  /** Stop capture, release the camera and the sensor listeners. */
  stop(): Promise<void>;

  /**
   * Pull the freshest pose. For renderers: you want the pose at *your* draw
   * time, not at every camera frame.
   *
   * @returns `null` before the first successful track.
   */
  currentPose(): Pose | null;

  /**
   * Push, fires once per tracked frame. For recorders, teleop transmitters,
   * anything that must not drop samples.
   */
  onPose(cb: (pose: Pose) => void): Unsubscribe;

  /** Tracking state transitions. */
  onState(cb: (state: TrackingState, prev: TrackingState) => void): Unsubscribe;

  /** Fires when the session recovers into the map after loss. */
  onRelocalize(cb: (event: RelocalizeEvent) => void): Unsubscribe;

  /** Fires for every loop-closure candidate, accepted or not. */
  onLoopClosure(cb: (event: LoopClosureEvent) => void): Unsubscribe;

  /** Current tracking state. */
  readonly state: TrackingState;

  /** Serialise the map. The caller stores the bytes. */
  saveMap(): Promise<Uint8Array>;

  /** The unstable debug surface. */
  readonly debug: WebSlamDebug;

  /** Version of the stable public API. */
  readonly version: string;
}
