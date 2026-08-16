/**
 * `web-slam/three` — camera-sync helper.
 *
 * spec.md §3: *"A `web-slam/three` entry point providing a camera-sync helper.
 * Adoption friction is mostly in the first fifteen minutes."*
 *
 * `three` is a **peer** dependency and is never imported here: this module is
 * typed structurally against the two or three properties it touches, so a
 * consumer on any three.js version works, and a consumer who does not use
 * three.js does not pay for this file.
 */

import type { Pose, TrackingState } from './types.js';
import { hasPose } from './types.js';

/** The subset of `THREE.Camera` this helper touches. */
export interface CameraLike {
  matrix: { fromArray(array: ArrayLike<number>): unknown };
  matrixAutoUpdate: boolean;
  projectionMatrix?: { makePerspective?: unknown };
  fov?: number;
  aspect?: number;
  near?: number;
  far?: number;
  updateProjectionMatrix?(): void;
}

/** Options for {@link CameraSync}. */
export interface CameraSyncOptions {
  /**
   * Keep the last good pose when tracking degrades, rather than freezing the
   * camera at the origin or snapping.
   *
   * Defaults to `true`. A renderer that jumps to identity on every brief
   * occlusion is worse than one that holds still for 300 ms.
   */
  holdOnLoss?: boolean;
  /**
   * Optional exponential smoothing factor in `(0, 1]`, applied to position
   * only. `1` (default) means no smoothing.
   *
   * Off by default deliberately: smoothing trades latency for stability, and
   * that is the consumer's call to make, not ours to make silently.
   */
  smoothing?: number;
}

/**
 * Drives a three.js camera from a web-slam session.
 *
 * ```ts
 * const sync = new CameraSync(camera);
 * renderLoop(() => {
 *   sync.update(slam.currentPose());
 *   renderer.render(scene, camera);
 * });
 * ```
 */
export class CameraSync {
  private readonly camera: CameraLike;
  private readonly holdOnLoss: boolean;
  private readonly smoothing: number;
  private smoothed: Float32Array | null = null;
  private readonly gl = new Float32Array(16);
  private lastState: TrackingState = 'initializing';

  constructor(camera: CameraLike, options: CameraSyncOptions = {}) {
    this.camera = camera;
    this.holdOnLoss = options.holdOnLoss ?? true;
    this.smoothing = options.smoothing ?? 1;
    if (!(this.smoothing > 0 && this.smoothing <= 1)) {
      throw new RangeError(
        `CameraSync: smoothing must be in (0, 1], got ${options.smoothing}`,
      );
    }
    // web-slam supplies the full world matrix; letting three.js recompose it
    // from position/quaternion/scale every frame would silently discard it.
    camera.matrixAutoUpdate = false;
  }

  /** Apply a pose. Passing `null` or a lost-state pose is safe. */
  update(pose: Pose | null): void {
    if (!pose) return;
    this.lastState = pose.state;
    if (!hasPose(pose.state) && this.holdOnLoss) return;

    // `pose.matrix` is `T_world_camera` in the tracker's computer-vision
    // convention: x right, **y down, z forward** — that is what `fx·x/z + cx`
    // projecting with z > 0 in front means. A three.js camera is the opposite
    // handed basis: y up, looking down −z. Applying the CV matrix raw renders
    // a view flipped 180° about x — tilt the phone up and the scene tilts
    // down, which reads as broken tracking rather than as the axis bug it is.
    //
    // Right-multiplying by diag(1,−1,−1,1) re-expresses the same physical
    // camera in GL axes: column-major, that is negating columns 1 and 2. The
    // world stays in CV coordinates, so world-space content (landmarks, the
    // trail) projects through this camera exactly as the tracker saw it.
    const cv = pose.matrix;
    const m = this.gl;
    m.set(cv);
    for (let i = 4; i < 12; i++) m[i] = -cv[i];

    if (this.smoothing < 1) {
      if (!this.smoothed) {
        this.smoothed = new Float32Array(m);
      } else {
        // Position only (indices 12..14 of a column-major 4x4). Smoothing the
        // rotation block element-wise would denormalise the basis.
        for (let i = 12; i < 15; i++) {
          this.smoothed[i] += (m[i] - this.smoothed[i]) * this.smoothing;
        }
        this.smoothed.set(m.subarray(0, 12), 0);
      }
      this.camera.matrix.fromArray(this.smoothed);
      return;
    }
    this.camera.matrix.fromArray(m);
  }

  /** State of the most recently applied pose. */
  get state(): TrackingState {
    return this.lastState;
  }

  /**
   * Set the camera's vertical FOV from web-slam's estimated intrinsics.
   *
   * Call after L2 converges — and again when it refines; it is idempotent.
   * Rendering with the default 50-degree three.js FOV over a 66-degree camera
   * feed produces a registration error that looks exactly like a tracking
   * bug, and is the single most common integration mistake.
   *
   * When the video is displayed with `object-fit: cover` (every full-bleed AR
   * layout), the pane shows only a centred crop of the image, so the pane's
   * FOV is narrower than the camera's. Pass the pane size as `viewport` and
   * the crop is accounted for; omit it for a renderer that shows the full
   * frame.
   */
  setIntrinsics(
    focalPx: number,
    imageWidth: number,
    imageHeight: number,
    viewport?: { width: number; height: number },
  ): void {
    if (!(focalPx > 0 && imageHeight > 0 && imageWidth > 0)) return;
    let visibleRows = imageHeight;
    let aspect = imageWidth / imageHeight;
    if (viewport && viewport.width > 0 && viewport.height > 0) {
      // Cover-fit: the image is scaled by max(paneW/W, paneH/H) and the
      // overflow is cropped equally on both sides. The rows that survive are
      // what the pane's vertical FOV subtends.
      const s = Math.max(viewport.width / imageWidth, viewport.height / imageHeight);
      visibleRows = viewport.height / s;
      aspect = viewport.width / viewport.height;
    }
    const vfov = 2 * Math.atan(visibleRows / 2 / focalPx) * (180 / Math.PI);
    this.camera.fov = vfov;
    this.camera.aspect = aspect;
    this.camera.updateProjectionMatrix?.();
  }
}

/**
 * Convert a 6x6 pose covariance into the three semi-axis lengths and the
 * orientation of a 1-sigma position uncertainty ellipsoid.
 *
 * spec.md §8 lists the covariance ellipsoid as a required dev-viewer element:
 * *"it is our differentiator, we should be looking at it daily."* Making it two
 * lines for a consumer is how it ends up on screen.
 *
 * @returns semi-axes in the same units as `pose.position`, plus the column-major
 * 3x3 rotation whose columns are the principal axes.
 */
export function positionUncertaintyEllipsoid(
  covariance: Float64Array,
  sigma = 1,
): { axes: [number, number, number]; basis: Float64Array } {
  // Extract the 3x3 translation block from the row-major 6x6.
  const a: number[][] = [
    [covariance[0], covariance[1], covariance[2]],
    [covariance[6], covariance[7], covariance[8]],
    [covariance[12], covariance[13], covariance[14]],
  ];
  // Jacobi eigenvalue iteration. Three sweeps is plenty for a 3x3 symmetric
  // matrix and avoids pulling in a linear-algebra dependency.
  const v: number[][] = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  for (let sweep = 0; sweep < 12; sweep++) {
    let off = 0;
    for (let p = 0; p < 3; p++)
      for (let q = p + 1; q < 3; q++) off += a[p][q] * a[p][q];
    if (off < 1e-24) break;
    for (let p = 0; p < 3; p++) {
      for (let q = p + 1; q < 3; q++) {
        if (Math.abs(a[p][q]) < 1e-30) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t =
          Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < 3; k++) {
          const akp = a[k][p];
          const akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < 3; k++) {
          const apk = a[p][k];
          const aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
          const vkp = v[k][p];
          const vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const axes: [number, number, number] = [
    sigma * Math.sqrt(Math.max(0, a[0][0])),
    sigma * Math.sqrt(Math.max(0, a[1][1])),
    sigma * Math.sqrt(Math.max(0, a[2][2])),
  ];
  const basis = new Float64Array([
    v[0][0], v[1][0], v[2][0],
    v[0][1], v[1][1], v[2][1],
    v[0][2], v[1][2], v[2][2],
  ]);
  return { axes, basis };
}
