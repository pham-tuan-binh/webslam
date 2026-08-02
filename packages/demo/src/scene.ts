/**
 * The three.js side of the demo.
 *
 * Deliberately decoupled from the compute path (spec.md §8: "Three.js is fine
 * here; it is decoupled from the compute path and not performance-critical").
 * It consumes only public and debug-surface types — it never imports the
 * backend, and it never touches a wasm module.
 */

import * as THREE from 'three';
import { positionUncertaintyEllipsoid } from 'web-slam/three';
import type { DebugKeyframe, Pose } from 'web-slam';

const MAX_LANDMARKS = 20_000;
const MAX_TRAJECTORY_POINTS = 12_000;

export class SceneView {
  readonly camera: THREE.PerspectiveCamera;
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

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: true,
    });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

    this.camera = new THREE.PerspectiveCamera(60, 1, 0.01, 100);
    // web-slam supplies the full world matrix; three must not recompose it.
    this.camera.matrixAutoUpdate = false;

    this.landmarkGeometry.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(MAX_LANDMARKS * 3), 3),
    );
    this.landmarkGeometry.setDrawRange(0, 0);
    this.landmarks = new THREE.Points(
      this.landmarkGeometry,
      new THREE.PointsMaterial({
        color: 0x5eead4,
        size: 0.018,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.85,
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
      new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55 }),
    );
    this.scene.add(this.trajectory);

    this.scene.add(this.frusta);

    // The covariance ellipsoid. spec.md §8 lists it as a required viewer
    // element — "it is our differentiator, we should be looking at it daily" —
    // and it belongs in the public demo for the same reason.
    this.uncertainty = new THREE.Mesh(
      new THREE.SphereGeometry(1, 24, 16),
      new THREE.MeshBasicMaterial({
        color: 0xfbbf24,
        wireframe: true,
        transparent: true,
        opacity: 0.4,
      }),
    );
    this.scene.add(this.uncertainty);

    this.resize();
    addEventListener('resize', () => this.resize());
  }

  private resize(): void {
    const w = innerWidth;
    const h = innerHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
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

  /** Brief visual confirmation that the session recovered into the map. */
  flashRelocalization(): void {
    this.relocFlash = 1;
  }

  render(): void {
    if (this.relocFlash > 0) {
      this.relocFlash = Math.max(0, this.relocFlash - 0.03);
      const material = this.landmarks.material as THREE.PointsMaterial;
      material.color.setHex(0xffffff).lerp(new THREE.Color(0x5eead4), 1 - this.relocFlash);
      material.size = 0.018 + this.relocFlash * 0.02;
    }
    this.renderer.render(this.scene, this.camera);
  }
}

/** A small wireframe camera frustum, drawn once per keyframe. */
function makeFrustum(scale = 0.06): THREE.LineSegments {
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
    new THREE.LineBasicMaterial({ color: 0x8b93a7, transparent: true, opacity: 0.5 }),
  );
}
