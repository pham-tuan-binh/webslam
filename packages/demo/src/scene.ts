/**
 * The three.js side of the demo.
 *
 * Deliberately decoupled from the compute path (spec.md §8: "Three.js is fine
 * here; it is decoupled from the compute path and not performance-critical").
 * It consumes only public and debug-surface types — it never imports the
 * backend, and it never touches a wasm module.
 *
 * One canvas, two viewports. The renderer scissors the screen into the AR
 * view (top half, transparent, composited over the camera feed) and the
 * trajectory view (bottom half, opaque). Objects that only make sense from
 * outside the reconstruction — the grid, the keyframe trail, the live
 * frustum, the uncertainty ellipsoid — live on `MAP_LAYER`, which only the
 * map camera renders. The landmark cloud and the trail render in both: in AR
 * they are the visual proof that tracking is anchored to the world.
 */

import * as THREE from 'three';
import { positionUncertaintyEllipsoid } from 'web-slam/three';
import type { DebugKeyframe, Pose } from 'web-slam';

const MAX_LANDMARKS = 20_000;
const MAX_TRAJECTORY_POINTS = 12_000;

/** Objects on this layer render only in the bottom (map) viewport. */
const MAP_LAYER = 1;

const TRAIL_COLOR = 0xe6e8eb;
const LIVE_COLOR = 0x6fb7e8;
const LANDMARK_COLOR = 0x3d4854;
const KEYFRAME_COLOR = 0x333d47;
const GRID_MAJOR = 0x181d22;
const GRID_MINOR = 0x10141a;
const MAP_CLEAR = 0x08090c;

export class SceneView {
  /** Driven by the pose estimate; renders the top viewport. */
  readonly camera: THREE.PerspectiveCamera;
  /** Free camera for the bottom viewport; orbits and follows the trail head. */
  readonly mapCamera: THREE.PerspectiveCamera;
  private readonly liveFrustum: THREE.Object3D;
  private readonly liveMark: THREE.LineSegments;
  private orbit = { theta: 0.9, phi: 1.15, radius: 5, target: new THREE.Vector3() };
  /** Where the trail currently ends; the orbit target eases toward it. */
  private readonly followTarget = new THREE.Vector3();
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();

  private readonly landmarks: THREE.Points;
  private readonly landmarkGeometry = new THREE.BufferGeometry();
  private readonly trajectory: THREE.Line;
  private readonly trajectoryGeometry = new THREE.BufferGeometry();
  private readonly frusta = new THREE.Group();
  private readonly uncertainty: THREE.Mesh;
  private relocFlash = 0;
  private keyframesDrawn = 0;
  private trailPoints = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: true,
    });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    // Two viewports per frame; each pass clears its own scissored region.
    this.renderer.autoClear = false;

    this.camera = new THREE.PerspectiveCamera(60, 1, 0.01, 100);
    // web-slam supplies the full world matrix; three must not recompose it.
    this.camera.matrixAutoUpdate = false;

    // The map camera is ordinary: three composes it from the orbit state. It
    // renders the default layer plus everything marked map-only.
    this.mapCamera = new THREE.PerspectiveCamera(55, 1, 0.01, 500);
    this.mapCamera.layers.enable(MAP_LAYER);

    // A metre grid. Without a fixed reference the trail has no sense of scale
    // or orientation and every reconstruction looks plausible. Kept recessive:
    // the data is the bright thing, the reference is barely there.
    const grid = new THREE.GridHelper(20, 40, GRID_MAJOR, GRID_MINOR);
    grid.layers.set(MAP_LAYER);
    this.scene.add(grid);
    // Origin marker. A muted cross, not an RGB axes helper — the only color
    // in the trajectory pane belongs to the data.
    const origin = makeCross(0x38424d, 0.3);
    origin.layers.set(MAP_LAYER);
    this.scene.add(origin);

    // The live pose, drawn the same way keyframes are so the two are directly
    // comparable: if the live frustum drifts away from the trail of keyframes,
    // that is the drift, visible.
    this.liveFrustum = makeFrustum(LIVE_COLOR, 0.14);
    this.liveFrustum.matrixAutoUpdate = false;
    this.liveFrustum.visible = false;
    this.liveFrustum.layers.set(MAP_LAYER);
    this.scene.add(this.liveFrustum);

    // A three-axis cross at the trail head — the "you are here" the frustum
    // alone does not give when it points away from the camera.
    this.liveMark = makeCross(LIVE_COLOR, 0.07);
    this.liveMark.visible = false;
    this.liveMark.layers.set(MAP_LAYER);
    this.scene.add(this.liveMark);

    this.landmarkGeometry.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(MAX_LANDMARKS * 3), 3),
    );
    this.landmarkGeometry.setDrawRange(0, 0);
    this.landmarks = new THREE.Points(
      this.landmarkGeometry,
      new THREE.PointsMaterial({
        color: LANDMARK_COLOR,
        size: 0.02,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.9,
        depthWrite: false,
      }),
    );
    this.scene.add(this.landmarks);

    this.trajectoryGeometry.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(MAX_TRAJECTORY_POINTS * 3), 3),
    );
    this.trajectoryGeometry.setDrawRange(0, 0);
    this.trajectory = new THREE.Line(
      this.trajectoryGeometry,
      new THREE.LineBasicMaterial({ color: TRAIL_COLOR, transparent: true, opacity: 0.95 }),
    );
    this.scene.add(this.trajectory);

    this.frusta.add(new THREE.Group());
    this.scene.add(this.frusta);

    // The covariance ellipsoid. spec.md §8 lists it as a required viewer
    // element — "it is our differentiator, we should be looking at it daily" —
    // and it belongs in the public demo for the same reason.
    this.uncertainty = new THREE.Mesh(
      new THREE.SphereGeometry(1, 24, 16),
      new THREE.MeshBasicMaterial({
        color: 0xd9a514,
        wireframe: true,
        transparent: true,
        opacity: 0.3,
      }),
    );
    // Hidden until the first pose arrives — before that there is no estimate
    // for it to be the uncertainty *of*, and a unit sphere at the origin
    // reads as data.
    this.uncertainty.visible = false;
    this.uncertainty.layers.set(MAP_LAYER);
    this.scene.add(this.uncertainty);

    this.resize();
    addEventListener('resize', () => this.resize());
  }

  private resize(): void {
    const w = innerWidth;
    const h = innerHeight;
    this.renderer.setSize(w, h, false);
    // Each camera sees one half of the screen.
    this.camera.aspect = w / (h / 2);
    this.camera.updateProjectionMatrix();
    this.mapCamera.aspect = w / (h / 2);
    this.mapCamera.updateProjectionMatrix();
  }

  setLandmarks(packed: Float32Array): void {
    const count = Math.min(packed.length / 3, MAX_LANDMARKS);
    const attr = this.landmarkGeometry.getAttribute('position') as THREE.BufferAttribute;
    (attr.array as Float32Array).set(packed.subarray(0, count * 3));
    attr.needsUpdate = true;
    this.landmarkGeometry.setDrawRange(0, count);
  }

  setTrajectory(packed: Float32Array): void {
    const count = Math.min(packed.length / 3, MAX_TRAJECTORY_POINTS);
    const attr = this.trajectoryGeometry.getAttribute('position') as THREE.BufferAttribute;
    (attr.array as Float32Array).set(packed.subarray(0, count * 3));
    attr.needsUpdate = true;
    this.trajectoryGeometry.setDrawRange(0, count);
    this.trailPoints = count;
    if (count > 0) {
      this.followTarget.set(
        packed[(count - 1) * 3],
        packed[(count - 1) * 3 + 1],
        packed[(count - 1) * 3 + 2],
      );
      this.liveMark.position.copy(this.followTarget);
      this.liveMark.visible = true;
    }
  }

  /**
   * Accumulate keyframe frusta as the user walks.
   *
   * Only new keyframes are added — rebuilding the group every frame would
   * allocate at 60 Hz for no visual gain, and the demo is judged partly on
   * whether it stays smooth on a phone.
   */
  setKeyframes(keyframes: DebugKeyframe[]): void {
    for (let i = this.keyframesDrawn; i < keyframes.length; i++) {
      const frustum = makeFrustum();
      frustum.matrixAutoUpdate = false;
      frustum.matrix.fromArray(keyframes[i].matrix);
      frustum.layers.set(MAP_LAYER);
      this.frusta.add(frustum);
    }
    this.keyframesDrawn = keyframes.length;
  }

  /** Size and place the 2-sigma position uncertainty ellipsoid. */
  setUncertainty(pose: Pose): void {
    const { axes, basis } = positionUncertaintyEllipsoid(pose.covariance, 2);
    const finite = axes.every((a) => Number.isFinite(a) && a > 0);
    this.uncertainty.visible = finite;
    if (!finite) return;
    this.uncertainty.position.set(pose.position.x, pose.position.y, pose.position.z);
    this.uncertainty.scale.set(axes[0], axes[1], axes[2]);
    const m = new THREE.Matrix4().set(
      basis[0], basis[3], basis[6], 0,
      basis[1], basis[4], basis[7], 0,
      basis[2], basis[5], basis[8], 0,
      0, 0, 0, 1,
    );
    this.uncertainty.quaternion.setFromRotationMatrix(m);
  }

  /** Place the live camera frustum from the current pose. */
  setLivePose(matrix: Float32Array): void {
    this.liveFrustum.matrix.fromArray(matrix);
    this.liveFrustum.visible = true;
  }

  /** Orbit the map camera. Angles in radians, radius multiplied. */
  orbitBy(dTheta: number, dPhi: number, dRadius = 1): void {
    this.orbit.theta += dTheta;
    // Stop just short of the poles, where the up vector flips and the view
    // snaps through itself.
    this.orbit.phi = Math.min(Math.PI - 0.05, Math.max(0.05, this.orbit.phi + dPhi));
    this.orbit.radius = Math.min(200, Math.max(0.2, this.orbit.radius * dRadius));
  }

  /** Cumulative trail length, in map units. */
  trailLength(): number {
    const a = this.trajectoryGeometry.getAttribute('position').array as Float32Array;
    let sum = 0;
    for (let i = 1; i < this.trailPoints; i++) {
      const dx = a[i * 3] - a[(i - 1) * 3];
      const dy = a[i * 3 + 1] - a[(i - 1) * 3 + 1];
      const dz = a[i * 3 + 2] - a[(i - 1) * 3 + 2];
      sum += Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
    return sum;
  }

  /** Brief visual confirmation that the session recovered into the map. */
  flashRelocalization(): void {
    this.relocFlash = 1;
  }

  render(): void {
    // The orbit target eases toward the trail head, so the bottom view is a
    // chase camera by default and a free orbit the moment the user drags.
    this.orbit.target.lerp(this.followTarget, 0.08);
    const { theta, phi, radius, target } = this.orbit;
    this.mapCamera.position.set(
      target.x + radius * Math.sin(phi) * Math.cos(theta),
      target.y + radius * Math.cos(phi),
      target.z + radius * Math.sin(phi) * Math.sin(theta),
    );
    this.mapCamera.lookAt(target);

    if (this.relocFlash > 0) {
      this.relocFlash = Math.max(0, this.relocFlash - 0.03);
      const material = this.landmarks.material as THREE.PointsMaterial;
      material.color.setHex(0xffffff).lerp(new THREE.Color(LANDMARK_COLOR), 1 - this.relocFlash);
    }

    const w = this.renderer.domElement.width;
    const h = this.renderer.domElement.height;
    const half = Math.floor(h / 2);
    this.renderer.setScissorTest(true);

    // Top: AR overlay, transparent over the camera feed.
    this.renderer.setViewport(0, h - half, w, half);
    this.renderer.setScissor(0, h - half, w, half);
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.clear();
    this.renderer.render(this.scene, this.camera);

    // Bottom: the trajectory view, opaque.
    this.renderer.setViewport(0, 0, w, h - half);
    this.renderer.setScissor(0, 0, w, h - half);
    this.renderer.setClearColor(MAP_CLEAR, 1);
    this.renderer.clear();
    this.renderer.render(this.scene, this.mapCamera);

    this.renderer.setScissorTest(false);
  }
}

/** A small wireframe camera frustum, drawn once per keyframe. */
function makeFrustum(color = KEYFRAME_COLOR, scale = 0.06): THREE.LineSegments {
  const d = scale;
  const w = scale * 0.8;
  const h = scale * 0.55;
  // Apex at the optical centre, four rays to the image corners, plus the
  // rectangle joining them — the standard way to read camera orientation at a
  // glance in a point cloud.
  // prettier-ignore
  const points = new Float32Array([
    0, 0, 0,  -w, -h, d,
    0, 0, 0,   w, -h, d,
    0, 0, 0,   w,  h, d,
    0, 0, 0,  -w,  h, d,
    -w, -h, d,  w, -h, d,
     w, -h, d,  w,  h, d,
     w,  h, d, -w,  h, d,
    -w,  h, d, -w, -h, d,
  ]);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(points, 3));
  return new THREE.LineSegments(
    geometry,
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9 }),
  );
}

/** A three-axis cross — the position marker at the head of the trail. */
function makeCross(color: number, r: number): THREE.LineSegments {
  // prettier-ignore
  const points = new Float32Array([
    -r, 0, 0,  r, 0, 0,
    0, -r, 0,  0, r, 0,
    0, 0, -r,  0, 0, r,
  ]);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(points, 3));
  return new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color }));
}
