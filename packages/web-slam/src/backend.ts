/**
 * The seam between the public API and whatever is computing poses.
 *
 * Two implementations: {@link MockBackend} (M0, synthetic) and the wasm
 * backend. spec.md §3 requires the demo to be written at M0 against the mock
 * *"to validate the API with a real consumer before any of it exists, and to
 * make the interface expensive to change afterwards rather than cheap to
 * drift."* That only works if the seam is here, below the public types.
 */

import type {
  DebugFeature,
  DebugKeyframe,
  DebugPoseGraphEdge,
  DebugTimings,
  LoopClosureEvent,
  Pose,
  RelocalizeEvent,
  ScaleSourceSpec,
  SensorTier,
  TrackingState,
} from './types.js';
import type { RawFrame, RawMotion } from './shim.js';

/** Anything the public `WebSlam` object can be driven by. */
export interface Backend {
  /** Configure before the first frame. */
  configure(options: BackendOptions): Promise<void>;
  /** Feed one camera frame, exactly as the shim delivered it. */
  pushFrame(frame: RawFrame): void;
  /** Feed one motion event, exactly as the shim delivered it. */
  pushMotion(motion: RawMotion): void;
  /**
   * Advance the pipeline and return any poses produced since the last call.
   *
   * Returning a list rather than a single pose keeps the push path honest: a
   * recorder subscribed via `onPose` must not lose samples because the host
   * happened to call at a slow rate (spec.md §3, pull vs push).
   */
  step(): StepResult;
  /** Serialise the map. */
  saveMap(): Promise<Uint8Array>;
  /**
   * The current focal-length estimate and the image size it is expressed in,
   * or `null` when the backend has no camera model (the mock) or has not been
   * configured yet.
   *
   * L2 refines this during the init pan, so consumers registering an AR
   * projection should re-read it rather than caching the first value — see
   * `CameraSync.setIntrinsics`.
   */
  intrinsicsEstimate(): { focalPx: number; width: number; height: number } | null;
  /** Release resources. */
  dispose(): void;
  /** The debug surface's data source. */
  readonly debug: BackendDebug;
}

/** Configuration handed to a backend once, before the first frame. */
export interface BackendOptions {
  scale: ScaleSourceSpec;
  tier: SensorTier;
  map: boolean;
  seed: number;
  focalPrior?: number;
  width: number;
  height: number;
  /** `false` when motion permission was denied — forces tier 1. */
  motionAvailable: boolean;
  /**
   * `screen.orientation.angle` at session start: degrees the screen is
   * rotated counter-clockwise from its natural orientation.
   *
   * The pipeline derives the camera↔IMU extrinsic from it. `DeviceMotion`
   * reports in the *device's* axes while the browser delivers camera frames
   * upright in the *viewport's*; the two differ by this angle plus the rear
   * camera's 180°-about-x mounting. Leaving the extrinsic at identity fed L2
   * rotations from the wrong frame and corrupted the focal estimate — the
   * −53.7% failure documented on `SlamConfig::body_from_camera`.
   */
  screenOrientationDeg: number;
}

/** What one `step()` produced. */
export interface StepResult {
  /** Poses produced since the previous step, oldest first. */
  poses: Pose[];
  /** State transitions, oldest first. */
  transitions: { from: TrackingState; to: TrackingState }[];
  relocalizations: RelocalizeEvent[];
  loopClosures: LoopClosureEvent[];
}

/** Empty result, so implementations do not each invent one. */
export function emptyStep(): StepResult {
  return { poses: [], transitions: [], relocalizations: [], loopClosures: [] };
}

/** Data source behind `slam.debug`. */
export interface BackendDebug {
  landmarks(): Float32Array;
  keyframes(): DebugKeyframe[];
  trajectory(): Float32Array;
  features(): DebugFeature[];
  poseGraph(): DebugPoseGraphEdge[];
  timings(): DebugTimings;
}

/** Zeroed timings, for backends between frames. */
export function zeroTimings(): DebugTimings {
  return { upload: 0, pyramid: 0, corners: 0, flow: 0, pnp: 0, total: 0 };
}
