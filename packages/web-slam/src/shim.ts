/**
 * The browser sensor shim.
 *
 * spec.md §7 states the review rule for this file, and it is a rule rather than
 * a style preference:
 *
 * > **Keep it deliberately stupid.** It stamps arrival at the earliest possible
 * > instant and passes raw bytes and raw timestamps into WASM. No buffering, no
 * > smoothing, no reordering, no processing of any kind. L0 exists to measure
 * > event-loop jitter; a shim that helpfully cleans up its inputs destroys the
 * > signal it is supposed to deliver.
 *
 * Concretely, what this file must never do:
 * - average, filter or low-pass any sensor value
 * - reorder or deduplicate events
 * - drop an event because it "looks wrong"
 * - substitute `Date.now()` when a native timestamp is missing
 * - hold a queue longer than one frame
 *
 * Everything it *does* do: acquire the stream, read the frame, stamp arrival,
 * hand the bytes on. Reviewers should be able to confirm that by reading it.
 */

/** A camera frame, exactly as the browser handed it to us. */
export interface RawFrame {
  /** Monotonic delivery index. Increments even when frames are dropped. */
  index: number;
  /**
   * `VideoFrameCallbackMetadata.mediaTime`, in seconds.
   *
   * This rides the media clock rather than the wall clock, which is the entire
   * reason we ask for it (spec.md §4 L0). Falls back to `presentationTime`
   * converted to seconds only when `mediaTime` is absent.
   */
  mediaTime: number;
  /** Arrival stamp from `performance.now()`, taken at the top of the callback. */
  arrivalMs: number;
  /** Tightly packed RGBA. */
  rgba: Uint8ClampedArray;
  width: number;
  height: number;
}

/** A motion event, converted to nothing, filtered by nothing. */
export interface RawMotion {
  /** Monotonic delivery index — what L0 fits its cadence model over. */
  index: number;
  /** Arrival stamp from `performance.now()`, taken at the top of the handler. */
  arrivalMs: number;
  /** `rotationRate` in **degrees per second**, as delivered. */
  gyroDegPerS: [number, number, number];
  /** `accelerationIncludingGravity` in m/s^2, as delivered. */
  accelWithGravity: [number, number, number];
  /** `acceleration` in m/s^2 if the platform supplied it. */
  accelLinear: [number, number, number] | null;
  /** `interval` in seconds. */
  intervalS: number;
}

/** Whether `DeviceMotionEvent.requestPermission` exists (iOS 13+). */
export function motionPermissionRequired(): boolean {
  return (
    typeof DeviceMotionEvent !== 'undefined' &&
    typeof (DeviceMotionEvent as unknown as { requestPermission?: unknown })
      .requestPermission === 'function'
  );
}

/**
 * Request motion permission.
 *
 * **Must be called from a user gesture on iOS.** This is a hard platform
 * constraint; there is no way to hide it, which is why `slam.start()` is
 * explicit in the public API (spec.md §3).
 */
export async function requestMotionPermission(): Promise<PermissionState> {
  if (!motionPermissionRequired()) {
    return typeof DeviceMotionEvent !== 'undefined' ? 'granted' : 'denied';
  }
  const request = (
    DeviceMotionEvent as unknown as {
      requestPermission: () => Promise<'granted' | 'denied'>;
    }
  ).requestPermission;
  try {
    return await request();
  } catch {
    // A rejected promise here means "not from a user gesture", which is a
    // denial from our point of view. Surfacing it as such lets the caller fall
    // back to tier 1 instead of failing the session.
    return 'denied';
  }
}

/** Default camera constraints: rear camera, 720p, 60 fps if offered. */
export function defaultCameraConstraints(): MediaTrackConstraints {
  return {
    facingMode: { ideal: 'environment' },
    width: { ideal: 1280 },
    height: { ideal: 720 },
    frameRate: { ideal: 60 },
  };
}

/**
 * Pulls frames from a `<video>` element via `requestVideoFrameCallback`.
 *
 * `rVFC` rather than `requestAnimationFrame`: it fires once per *decoded* frame
 * and carries `mediaTime`, so we neither resample the camera onto the display
 * cadence nor lose the media-clock timestamp.
 */
export class CameraShim {
  private readonly video: HTMLVideoElement;
  private canvas: OffscreenCanvas | HTMLCanvasElement | null = null;
  private ctx:
    | OffscreenCanvasRenderingContext2D
    | CanvasRenderingContext2D
    | null = null;
  private stream: MediaStream | null = null;
  private handle: number | null = null;
  private index = 0;
  private running = false;
  private onFrame: ((frame: RawFrame) => void) | null = null;
  /**
   * Longest edge, in pixels, that frames are scaled to before tracking.
   *
   * A phone camera hands us 1280x720 or larger, which is 2.5x the pixel count
   * the tracker was tuned at and costs that much in the pyramid, corner and
   * flow stages — the three that dominate the frame budget. Nothing in
   * monocular SLAM benefits from that resolution: corner localisation is
   * sub-pixel either way. Scaling happens inside `drawImage`, so it is free
   * relative to the `getImageData` it shrinks.
   */
  private targetLongEdge = 640;
  /** The size frames are actually delivered at, after scaling. */
  processingSize: { width: number; height: number } | null = null;

  constructor(video: HTMLVideoElement) {
    this.video = video;
  }

  /** Whether the platform can deliver per-decoded-frame callbacks. */
  static supported(): boolean {
    return (
      typeof HTMLVideoElement !== 'undefined' &&
      'requestVideoFrameCallback' in HTMLVideoElement.prototype
    );
  }

  /** Acquire the camera and attach it to the video element. */
  async open(constraints?: MediaTrackConstraints): Promise<MediaStreamTrack> {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('web-slam: getUserMedia is unavailable in this context');
    }
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { ...defaultCameraConstraints(), ...constraints },
      audio: false,
    });
    this.video.srcObject = this.stream;
    this.video.playsInline = true;
    this.video.muted = true;
    await this.video.play();
    const [track] = this.stream.getVideoTracks();
    if (!track) throw new Error('web-slam: camera stream has no video track');
    return track;
  }

  /**
   * The size frames will be delivered at, given the camera's native size.
   *
   * The caller needs this *before* the first frame: intrinsics are derived
   * from the image dimensions, and configuring the pipeline for 1280x720 while
   * feeding it 640x360 puts the principal point and focal guess out by a
   * factor of two, which no bootstrap survives.
   */
  processingDimensions(): { width: number; height: number } {
    const size = scaleToLongEdge(this.video.videoWidth, this.video.videoHeight, this.targetLongEdge);
    this.processingSize = size;
    return size;
  }

  /** Begin delivering frames. */
  start(onFrame: (frame: RawFrame) => void): void {
    if (this.running) return;
    this.onFrame = onFrame;
    this.running = true;
    this.schedule();
  }

  private schedule(): void {
    if (!this.running) return;
    const video = this.video as HTMLVideoElement & {
      requestVideoFrameCallback?: (
        cb: (now: number, metadata: { mediaTime?: number; presentationTime?: number }) => void,
      ) => number;
    };
    if (typeof video.requestVideoFrameCallback === 'function') {
      this.handle = video.requestVideoFrameCallback((now, metadata) =>
        this.deliver(now, metadata),
      );
    } else {
      // rAF fallback. Documented as degraded: without mediaTime the clock layer
      // loses the media-clock reference and falls back to arrival stamps, which
      // is precisely the jitter L0 exists to remove.
      this.handle = requestAnimationFrame((now) => this.deliver(now, {}));
    }
  }

  private deliver(
    now: number,
    metadata: { mediaTime?: number; presentationTime?: number },
  ): void {
    // Stamp first. Everything after this line costs time we would otherwise
    // attribute to the sensor.
    const arrivalMs = performance.now();

    const nativeW = this.video.videoWidth;
    const nativeH = this.video.videoHeight;
    const { width, height } = scaleToLongEdge(nativeW, nativeH, this.targetLongEdge);
    if (width > 0 && height > 0 && this.onFrame) {
      const ctx = this.context(width, height);
      if (ctx) {
        // Scale during the blit. The destination rect is the processing size,
        // so getImageData reads the smaller buffer.
        ctx.drawImage(this.video as CanvasImageSource, 0, 0, width, height);
        const image = ctx.getImageData(0, 0, width, height);
        this.onFrame({
          index: this.index++,
          mediaTime:
            metadata.mediaTime ??
            (metadata.presentationTime !== undefined
              ? metadata.presentationTime / 1000
              : now / 1000),
          arrivalMs,
          rgba: image.data,
          width,
          height,
        });
      }
    }
    this.schedule();
  }

  private context(
    width: number,
    height: number,
  ): OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D | null {
    if (!this.canvas || this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas =
        typeof OffscreenCanvas !== 'undefined'
          ? new OffscreenCanvas(width, height)
          : Object.assign(document.createElement('canvas'), { width, height });
      this.canvas.width = width;
      this.canvas.height = height;
      // `willReadFrequently` matters: without it Safari keeps the canvas on the
      // GPU and every getImageData round-trips, which shows up directly in the
      // upload stage of the frame budget.
      this.ctx = (this.canvas as HTMLCanvasElement).getContext('2d', {
        willReadFrequently: true,
      }) as CanvasRenderingContext2D | null;
    }
    return this.ctx;
  }

  /** Stop delivering frames and release the camera. */
  stop(): void {
    this.running = false;
    if (this.handle !== null) {
      const video = this.video as HTMLVideoElement & {
        cancelVideoFrameCallback?: (handle: number) => void;
      };
      if (typeof video.cancelVideoFrameCallback === 'function') {
        video.cancelVideoFrameCallback(this.handle);
      } else {
        cancelAnimationFrame(this.handle);
      }
      this.handle = null;
    }
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.video.srcObject = null;
    this.onFrame = null;
  }
}

/**
 * Fit `(w, h)` inside `longEdge` without changing aspect ratio.
 *
 * Never upscales: a camera that already hands us something small is giving us
 * all the information there is, and inventing pixels would cost time without
 * adding a single corner.
 */
export function scaleToLongEdge(
  w: number,
  h: number,
  longEdge: number,
): { width: number; height: number } {
  if (w <= 0 || h <= 0) return { width: 0, height: 0 };
  const longest = Math.max(w, h);
  if (longest <= longEdge) return { width: w, height: h };
  const k = longEdge / longest;
  // Even dimensions keep the pyramid's halving exact at the first level.
  const even = (v: number) => Math.max(2, Math.round(v * k / 2) * 2);
  return { width: even(w), height: even(h) };
}

/** Forwards `devicemotion` events, unmodified. */
export class MotionShim {
  private index = 0;
  private listener: ((e: DeviceMotionEvent) => void) | null = null;
  private granted = false;

  /** Whether motion samples are flowing. `false` means tier 1. */
  get available(): boolean {
    return this.granted;
  }

  /** Request permission and begin listening. Call from a user gesture. */
  async start(onMotion: (motion: RawMotion) => void): Promise<boolean> {
    const permission = await requestMotionPermission();
    if (permission !== 'granted') {
      this.granted = false;
      return false;
    }
    this.listener = (e: DeviceMotionEvent) => {
      // Stamp first, before touching the event.
      const arrivalMs = performance.now();
      const r = e.rotationRate;
      const ag = e.accelerationIncludingGravity;
      const a = e.acceleration;
      // A sample missing both rate and gravity carries no information; passing
      // it on would inject a fabricated zero into the cadence model, which is
      // worse than the gap. This is a completeness check, not a filter — no
      // sample with real values is ever dropped.
      if (!r && !ag) return;
      onMotion({
        index: this.index++,
        arrivalMs,
        gyroDegPerS: [r?.beta ?? 0, r?.gamma ?? 0, r?.alpha ?? 0],
        accelWithGravity: [ag?.x ?? 0, ag?.y ?? 0, ag?.z ?? 0],
        accelLinear: a && a.x !== null ? [a.x ?? 0, a.y ?? 0, a.z ?? 0] : null,
        intervalS: (e.interval ?? 0) / 1000,
      });
    };
    window.addEventListener('devicemotion', this.listener, { passive: true });
    this.granted = true;
    return true;
  }

  /** Stop listening. */
  stop(): void {
    if (this.listener) {
      window.removeEventListener('devicemotion', this.listener);
      this.listener = null;
    }
    this.granted = false;
  }
}
