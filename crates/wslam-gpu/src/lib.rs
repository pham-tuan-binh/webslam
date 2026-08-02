//! # wslam-gpu
//!
//! The L3 image front-end on the GPU: upload, pyramid, Shi-Tomasi corners,
//! pyramidal Lucas-Kanade flow. One WGSL codebase runs natively through
//! Metal/Vulkan/DX12 and in the browser through WebGPU, which is the whole
//! reason spec.md §7 picked `wgpu` — the replay harness and the arm-rig
//! regression suite execute the same shaders that ship.
//!
//! ## Readback discipline
//!
//! spec.md §4 L3: *"read back a few hundred points, never a full image."* The
//! steady-state path honours that literally. The pyramid and the corner
//! response image are written, read and consumed entirely in device memory;
//! [`ImagePipeline::detect_corners`] returns one candidate per grid cell (9.6 kB
//! for a 640x480 frame) and [`ImagePipeline::track_flow`] returns 16 bytes per
//! tracked point. Nothing else crosses the bus.
//!
//! ## Determinism
//!
//! There is no clock and no RNG in this crate — nothing here needs either. What
//! does need pinning down is float summation order, because it is not
//! associative and the GPU is free to schedule however it likes. Every
//! reduction in this crate has a fixed, documented order, and
//! [`reference`] reproduces it. Ties in the corner non-max suppression resolve
//! to the lowest raster index rather than to whichever invocation got there
//! first.
//!
//! ## The reference is load-bearing
//!
//! spec.md §6 L3 requires that *"any divergence is a port bug, not an algorithm
//! result."* [`reference`] is a complete CPU implementation of every kernel
//! here, and the equivalence tests are what turn that sentence into something
//! falsifiable. A kernel without a reference is not finished.
//!
//! ```no_run
//! # async fn demo() -> wslam_core::Result<()> {
//! use wslam_gpu::{GpuContext, ImagePipeline, FlowConfig};
//! # let (frame_a, frame_b): (wslam_core::GrayImage, wslam_core::GrayImage) = unimplemented!();
//! let ctx = GpuContext::new().await?;
//! let mut pipe = ImagePipeline::new(&ctx, 640, 480, 4)?;
//!
//! pipe.upload(&frame_a)?;
//! pipe.build_pyramid()?;
//! let corners = pipe.detect_corners(300, 0.01, 8.0)?;
//!
//! pipe.swap();
//! pipe.upload(&frame_b)?;
//! pipe.build_pyramid()?;
//! let points: Vec<(f32, f32)> = corners.iter().map(|&(x, y, _)| (x, y)).collect();
//! let flow = pipe.track_flow(&points, &FlowConfig::default())?;
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]
#![deny(unsafe_code)]

pub mod reference;

#[cfg(test)]
mod testdata;

use bytemuck::{Pod, Zeroable};
use wslam_core::{Error, GrayImage, Result};

use reference::{CellWinner, CORNER_CELL_PX, MIN_STRUCTURE_EIGENVALUE};

const GRAYSCALE_WGSL: &str = include_str!("shaders/grayscale.wgsl");
const PYRAMID_WGSL: &str = include_str!("shaders/pyramid.wgsl");
const CORNERS_WGSL: &str = include_str!("shaders/corners.wgsl");
const KLT_WGSL: &str = include_str!("shaders/klt.wgsl");

/// Largest point list [`ImagePipeline::track_flow`] accepts in one call.
///
/// The flow buffers are sized once at construction rather than regrown per
/// frame; a tracker that wants more than this many features has a different
/// problem than throughput.
pub const MAX_TRACKED_POINTS: usize = 4096;

/// Largest supported Lucas-Kanade window. Odd, and bounded because the window
/// sum is serial within a workgroup.
pub const MAX_FLOW_WINDOW: u32 = 31;

/// Sentinel for "no pixel in this cell scored above zero", matching
/// `NO_WINNER` in `corners.wgsl`.
const NO_WINNER: u32 = u32::MAX;

/// Workgroup width shared by the two 1D kernels.
const WG_1D: u32 = 64;
/// Workgroup extent shared by the image-domain kernels.
const WG_2D: u32 = 8;

// ---------------------------------------------------------------------------
// Uniform block layouts. Field order and padding must match the WGSL structs.
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GrayParams {
    width: u32,
    height: u32,
    dst_offset: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PyramidParams {
    src_offset: u32,
    src_w: u32,
    src_h: u32,
    dst_offset: u32,
    dst_w: u32,
    dst_h: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CornerParams {
    offset: u32,
    width: u32,
    height: u32,
    cell_px: u32,
    grid_w: u32,
    grid_h: u32,
    cell_count: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct KltParams {
    levels: u32,
    window: u32,
    iterations: u32,
    point_count: u32,
    epsilon: f32,
    max_error: f32,
    min_eigenvalue: f32,
    border_margin: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LevelDesc {
    offset: u32,
    width: u32,
    height: u32,
    _pad: u32,
}

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

/// The device limits the tracker actually has to reason about.
///
/// A deliberately narrow view of `wgpu::Limits`: the two numbers that decide
/// whether a given frame size and feature count fit. Everything else in the
/// WebGPU limit set is either irrelevant to compute-only work or already
/// guaranteed by the baseline that spec.md §9 pins to iOS 26.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuLimits {
    /// Maximum invocations in a single compute workgroup.
    pub max_workgroup_size: u32,
    /// Maximum size of a single buffer allocation, in bytes.
    pub max_buffer_bytes: u64,
}

/// An open connection to a compute device.
///
/// Holds the `wgpu` device and queue and nothing else; the pipeline objects own
/// their own resources so that several can coexist (multi-resolution tracking,
/// or a replay harness running two configurations side by side).
#[derive(Debug)]
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    limits: GpuLimits,
    adapter_name: String,
}

impl GpuContext {
    /// Acquire an adapter and open a device.
    ///
    /// Requests only the WebGPU baseline limits, so a device that opens here
    /// opens in the browser too. Returns [`Error::Gpu`] when no adapter is
    /// available — callers in a test harness are expected to skip rather than
    /// fail, since headless CI machines frequently have none.
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(
            // Honour WGPU_BACKEND so a CI run can pin the backend it means to
            // exercise instead of taking whatever the machine offers.
            wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
        );
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|e| Error::Gpu(format!("no compute adapter available: {e}")))?;

        let info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("wslam-gpu"),
                required_features: wgpu::Features::empty(),
                // The WebGPU baseline. Asking for more here would open a device
                // natively that the browser build could not.
                required_limits: wgpu::Limits::defaults(),
                ..Default::default()
            })
            .await
            .map_err(|e| Error::Gpu(format!("device request failed: {e}")))?;

        let l = device.limits();
        log::debug!(
            "wslam-gpu: {} ({:?}, {:?})",
            info.name,
            info.backend,
            info.device_type
        );
        Ok(GpuContext {
            device,
            queue,
            limits: GpuLimits {
                max_workgroup_size: l.max_compute_invocations_per_workgroup,
                max_buffer_bytes: l.max_buffer_size,
            },
            adapter_name: info.name,
        })
    }

    /// Blocking wrapper around [`GpuContext::new`], for native harnesses and
    /// tests. Not available on wasm, where blocking the event loop deadlocks.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_blocking() -> Result<Self> {
        pollster::block_on(Self::new())
    }

    /// Device limits relevant to the tracker.
    #[must_use]
    pub fn limits(&self) -> GpuLimits {
        self.limits
    }

    /// Adapter name, for the per-device reporting spec.md §6 asks for
    /// ("Report per device. Aggregating hides the failure cases").
    #[must_use]
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }
}

// ---------------------------------------------------------------------------
// Flow configuration
// ---------------------------------------------------------------------------

/// Lucas-Kanade tuning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlowConfig {
    /// Side of the square tracking window in pixels. Must be odd and in
    /// `3..=`[`MAX_FLOW_WINDOW`].
    pub window: u32,
    /// Maximum Gauss-Newton iterations per pyramid level.
    pub iterations: u32,
    /// Convergence threshold on the per-iteration update, in pixels.
    pub epsilon: f32,
    /// Maximum mean absolute photometric residual, in grey levels, for a track
    /// to be reported as `ok`.
    pub max_error: f32,
}

impl Default for FlowConfig {
    fn default() -> Self {
        FlowConfig {
            window: 15,
            iterations: 30,
            epsilon: 0.01,
            max_error: 25.0,
        }
    }
}

impl FlowConfig {
    fn validate(&self) -> Result<()> {
        if self.window < 3 || self.window > MAX_FLOW_WINDOW {
            return Err(Error::Config(format!(
                "flow window {} outside 3..={MAX_FLOW_WINDOW}",
                self.window
            )));
        }
        if self.window % 2 == 0 {
            return Err(Error::Config(format!(
                "flow window must be odd, got {}",
                self.window
            )));
        }
        if self.iterations == 0 {
            return Err(Error::Config("flow iterations must be >= 1".into()));
        }
        if !(self.epsilon.is_finite() && self.epsilon >= 0.0) {
            return Err(Error::Config(format!(
                "flow epsilon must be finite and non-negative, got {}",
                self.epsilon
            )));
        }
        if !(self.max_error.is_finite() && self.max_error >= 0.0) {
            return Err(Error::Config(format!(
                "flow max_error must be finite and non-negative, got {}",
                self.max_error
            )));
        }
        Ok(())
    }
}

/// One tracked point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlowResult {
    /// Tracked x in the current frame, level-0 pixels.
    pub x: f32,
    /// Tracked y in the current frame, level-0 pixels.
    pub y: f32,
    /// Whether the track is usable. False means the system was rank deficient,
    /// the point left the supported region, or the residual exceeded
    /// [`FlowConfig::max_error`].
    pub ok: bool,
    /// Mean absolute photometric residual over the window, in grey levels,
    /// measured at the returned position.
    pub error: f32,
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Per-frame GPU state: one pyramid allocation and the bind groups that read it.
struct FrameSet {
    /// Owned so the allocation's lifetime is the pipeline's rather than that of
    /// whichever bind group happens to still reference it. Only the test
    /// readback hook reads the handle back out.
    #[cfg_attr(not(test), allow(dead_code))]
    pyramid: wgpu::Buffer,
    gray_bind: wgpu::BindGroup,
    pyramid_binds: Vec<wgpu::BindGroup>,
    corner_bind: wgpu::BindGroup,
    /// Tracks *from* the other set *to* this one.
    klt_bind: wgpu::BindGroup,
}

/// Full L3 image front-end on the GPU: upload -> pyramid -> corners -> flow.
///
/// Owns two frame sets and alternates between them with [`ImagePipeline::swap`],
/// so the previous frame's pyramid is still resident when flow runs. Every
/// buffer, pipeline and bind group is allocated once at construction; the
/// steady-state path writes uniforms, dispatches, and reads back a small list.
pub struct ImagePipeline {
    device: wgpu::Device,
    queue: wgpu::Queue,

    width: u32,
    height: u32,
    levels: u32,
    level_desc: Vec<LevelDesc>,
    grid_w: u32,
    grid_h: u32,

    unpack_luma: wgpu::ComputePipeline,
    rgba_to_luma: wgpu::ComputePipeline,
    downsample: wgpu::ComputePipeline,
    clear_cells: wgpu::ComputePipeline,
    shi_tomasi: wgpu::ComputePipeline,
    cell_reduce: wgpu::ComputePipeline,
    cell_pick: wgpu::ComputePipeline,
    track: wgpu::ComputePipeline,

    upload_buf: wgpu::Buffer,
    /// See [`FrameSet::pyramid`]: owned for lifetime, read only by the
    /// equivalence test.
    #[cfg_attr(not(test), allow(dead_code))]
    response_buf: wgpu::Buffer,
    cell_max_buf: wgpu::Buffer,
    cell_arg_buf: wgpu::Buffer,
    corner_readback: wgpu::Buffer,
    points_buf: wgpu::Buffer,
    results_buf: wgpu::Buffer,
    flow_readback: wgpu::Buffer,
    klt_params_buf: wgpu::Buffer,

    sets: [FrameSet; 2],
    current: usize,

    /// Reused so the per-frame path does not allocate. The luma plane is padded
    /// to a whole number of words because `write_buffer` copies in multiples of
    /// four bytes.
    upload_scratch: Vec<u8>,
}

impl ImagePipeline {
    /// Build the pipeline for a fixed frame size and pyramid depth.
    ///
    /// Level `k` has dimensions `max(width >> k, 1) x max(height >> k, 1)`.
    ///
    /// # Errors
    /// [`Error::Config`] for a degenerate size or level count, [`Error::Gpu`]
    /// if the resources would exceed the device's buffer limits.
    pub fn new(ctx: &GpuContext, width: u32, height: u32, levels: u32) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::Config(format!(
                "image size {width}x{height} has a zero dimension"
            )));
        }
        if levels == 0 || levels > 16 {
            return Err(Error::Config(format!(
                "pyramid levels must be in 1..=16, got {levels}"
            )));
        }

        let device = ctx.device.clone();
        let queue = ctx.queue.clone();

        // -- level table -----------------------------------------------------
        let dims = reference::pyramid_dims(width, height, levels);
        let mut level_desc = Vec::with_capacity(levels as usize);
        let mut offset = 0u32;
        for &(w, h) in &dims {
            level_desc.push(LevelDesc {
                offset,
                width: w,
                height: h,
                _pad: 0,
            });
            offset += w * h;
        }
        let pyramid_elems = offset as u64;
        let pyramid_bytes = pyramid_elems * 4;

        let grid_w = width.div_ceil(CORNER_CELL_PX);
        let grid_h = height.div_ceil(CORNER_CELL_PX);
        let cell_count = (grid_w * grid_h) as u64;

        let upload_bytes = (width as u64) * (height as u64) * 4;
        let largest = pyramid_bytes.max(upload_bytes);
        if largest > ctx.limits.max_buffer_bytes {
            return Err(Error::Gpu(format!(
                "{width}x{height} x {levels} levels needs a {largest} byte buffer, \
                 device allows {}",
                ctx.limits.max_buffer_bytes
            )));
        }
        if ctx.limits.max_workgroup_size < WG_1D {
            return Err(Error::Gpu(format!(
                "device workgroup limit {} is below the {WG_1D} this crate's kernels use",
                ctx.limits.max_workgroup_size
            )));
        }

        // -- shaders ---------------------------------------------------------
        let gray_mod = shader(&device, "grayscale", GRAYSCALE_WGSL);
        let pyr_mod = shader(&device, "pyramid", PYRAMID_WGSL);
        let corner_mod = shader(&device, "corners", CORNERS_WGSL);
        let klt_mod = shader(&device, "klt", KLT_WGSL);

        let gray_layout = bind_layout(
            &device,
            "gray",
            &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, false),
            ],
        );
        let pyr_layout = bind_layout(
            &device,
            "pyramid",
            &[uniform_entry(0), storage_entry(1, false)],
        );
        let corner_layout = bind_layout(
            &device,
            "corners",
            &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, false),
                storage_entry(4, false),
            ],
        );
        let klt_layout = bind_layout(
            &device,
            "klt",
            &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, false),
            ],
        );

        let unpack_luma = pipeline(
            &device,
            "unpack_luma",
            &gray_mod,
            &gray_layout,
            "unpack_luma",
        );
        let rgba_to_luma = pipeline(
            &device,
            "rgba_to_luma",
            &gray_mod,
            &gray_layout,
            "rgba_to_luma",
        );
        let downsample = pipeline(
            &device,
            "downsample",
            &pyr_mod,
            &pyr_layout,
            "downsample_2x2",
        );
        let clear_cells = pipeline(
            &device,
            "clear_cells",
            &corner_mod,
            &corner_layout,
            "clear_cells",
        );
        let shi_tomasi = pipeline(
            &device,
            "shi_tomasi",
            &corner_mod,
            &corner_layout,
            "shi_tomasi",
        );
        let cell_reduce = pipeline(
            &device,
            "cell_reduce",
            &corner_mod,
            &corner_layout,
            "cell_reduce",
        );
        let cell_pick = pipeline(
            &device,
            "cell_pick",
            &corner_mod,
            &corner_layout,
            "cell_pick",
        );
        let track = pipeline(&device, "klt_track", &klt_mod, &klt_layout, "track");

        // -- buffers ---------------------------------------------------------
        use wgpu::BufferUsages as U;
        let upload_buf = buffer(&device, "upload", upload_bytes, U::STORAGE | U::COPY_DST);
        // The pyramid and the response image are copyable only in a test build.
        // spec.md §4 L3 forbids reading a full image back on the frame path, and
        // withholding the usage flag makes that a validation error rather than a
        // convention someone can quietly break.
        let debug_copy = if cfg!(test) { U::COPY_SRC } else { U::empty() };
        let response_buf = buffer(
            &device,
            "response",
            (width as u64) * (height as u64) * 4,
            U::STORAGE | debug_copy,
        );
        let cell_max_buf = buffer(
            &device,
            "cell_max",
            cell_count * 4,
            U::STORAGE | U::COPY_SRC,
        );
        let cell_arg_buf = buffer(
            &device,
            "cell_arg",
            cell_count * 4,
            U::STORAGE | U::COPY_SRC,
        );
        let corner_readback = buffer(
            &device,
            "corner_readback",
            cell_count * 8,
            U::MAP_READ | U::COPY_DST,
        );
        let points_buf = buffer(
            &device,
            "points",
            (MAX_TRACKED_POINTS as u64) * 8,
            U::STORAGE | U::COPY_DST,
        );
        let results_buf = buffer(
            &device,
            "results",
            (MAX_TRACKED_POINTS as u64) * 16,
            U::STORAGE | U::COPY_SRC,
        );
        let flow_readback = buffer(
            &device,
            "flow_readback",
            (MAX_TRACKED_POINTS as u64) * 16,
            U::MAP_READ | U::COPY_DST,
        );

        let level_desc_buf = buffer(
            &device,
            "levels",
            (level_desc.len() as u64) * 16,
            U::STORAGE | U::COPY_DST,
        );
        queue.write_buffer(&level_desc_buf, 0, bytemuck::cast_slice(&level_desc));

        // Uniforms that never change are written once here; only the flow block
        // is touched per frame.
        let gray_params_buf = uniform(
            &device,
            &queue,
            "gray_params",
            &[GrayParams {
                width,
                height,
                dst_offset: 0,
                _pad: 0,
            }],
        );
        let pyr_params_bufs: Vec<wgpu::Buffer> = (1..levels as usize)
            .map(|k| {
                let s = level_desc[k - 1];
                let d = level_desc[k];
                uniform(
                    &device,
                    &queue,
                    "pyramid_params",
                    &[PyramidParams {
                        src_offset: s.offset,
                        src_w: s.width,
                        src_h: s.height,
                        dst_offset: d.offset,
                        dst_w: d.width,
                        dst_h: d.height,
                        _pad0: 0,
                        _pad1: 0,
                    }],
                )
            })
            .collect();
        let corner_params_buf = uniform(
            &device,
            &queue,
            "corner_params",
            &[CornerParams {
                offset: 0,
                width,
                height,
                cell_px: CORNER_CELL_PX,
                grid_w,
                grid_h,
                cell_count: grid_w * grid_h,
                _pad: 0,
            }],
        );
        let klt_params_buf = buffer(
            &device,
            "klt_params",
            std::mem::size_of::<KltParams>() as u64,
            U::UNIFORM | U::COPY_DST,
        );

        let pyramids: Vec<wgpu::Buffer> = (0..2)
            .map(|i| {
                buffer(
                    &device,
                    &format!("pyramid{i}"),
                    pyramid_bytes,
                    U::STORAGE | debug_copy,
                )
            })
            .collect();

        let make_set = |i: usize| FrameSet {
            pyramid: pyramids[i].clone(),
            gray_bind: bind_group(
                &device,
                "gray",
                &gray_layout,
                &[
                    gray_params_buf.as_entire_binding(),
                    upload_buf.as_entire_binding(),
                    pyramids[i].as_entire_binding(),
                ],
            ),
            pyramid_binds: pyr_params_bufs
                .iter()
                .map(|p| {
                    bind_group(
                        &device,
                        "pyramid",
                        &pyr_layout,
                        &[p.as_entire_binding(), pyramids[i].as_entire_binding()],
                    )
                })
                .collect(),
            corner_bind: bind_group(
                &device,
                "corners",
                &corner_layout,
                &[
                    corner_params_buf.as_entire_binding(),
                    pyramids[i].as_entire_binding(),
                    response_buf.as_entire_binding(),
                    cell_max_buf.as_entire_binding(),
                    cell_arg_buf.as_entire_binding(),
                ],
            ),
            klt_bind: bind_group(
                &device,
                "klt",
                &klt_layout,
                &[
                    klt_params_buf.as_entire_binding(),
                    // Previous frame is the template, current frame is the target.
                    pyramids[1 - i].as_entire_binding(),
                    pyramids[i].as_entire_binding(),
                    level_desc_buf.as_entire_binding(),
                    points_buf.as_entire_binding(),
                    results_buf.as_entire_binding(),
                ],
            ),
        };
        let sets = [make_set(0), make_set(1)];

        Ok(ImagePipeline {
            device,
            queue,
            width,
            height,
            levels,
            level_desc,
            grid_w,
            grid_h,
            unpack_luma,
            rgba_to_luma,
            downsample,
            clear_cells,
            shi_tomasi,
            cell_reduce,
            cell_pick,
            track,
            upload_buf,
            response_buf,
            cell_max_buf,
            cell_arg_buf,
            corner_readback,
            points_buf,
            results_buf,
            flow_readback,
            klt_params_buf,
            sets,
            current: 0,
            upload_scratch: Vec::new(),
        })
    }

    /// Frame width this pipeline was built for.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Frame height this pipeline was built for.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Pyramid depth.
    #[must_use]
    pub fn levels(&self) -> u32 {
        self.levels
    }

    /// Dimensions of pyramid level `k`, or `None` if the level does not exist.
    #[must_use]
    pub fn level_size(&self, k: u32) -> Option<(u32, u32)> {
        self.level_desc.get(k as usize).map(|d| (d.width, d.height))
    }

    /// Upload an 8-bit luma frame into the current set and expand it to level 0.
    ///
    /// # Errors
    /// [`Error::Config`] if the image size does not match the pipeline.
    pub fn upload(&mut self, image: &GrayImage) -> Result<()> {
        if image.width() != self.width || image.height() != self.height {
            return Err(Error::Config(format!(
                "frame is {}x{}, pipeline was built for {}x{}",
                image.width(),
                image.height(),
                self.width,
                self.height
            )));
        }
        // write_buffer copies in whole words; pad the tail rather than
        // shortening the copy, so the last partial word is well defined.
        let padded = image.data().len().next_multiple_of(4);
        self.upload_scratch.clear();
        self.upload_scratch.extend_from_slice(image.data());
        self.upload_scratch.resize(padded, 0);
        self.queue
            .write_buffer(&self.upload_buf, 0, &self.upload_scratch);
        self.dispatch_upload("upload-luma", true);
        Ok(())
    }

    /// Upload a tightly-packed RGBA8 frame and convert it to luma on the GPU.
    ///
    /// This is the browser front door: a canvas or video frame arrives as RGBA,
    /// and converting it here saves a full-image CPU pass immediately in front
    /// of the upload it feeds. The weights match
    /// [`GrayImage::from_rgba`] exactly, so the two paths produce identical
    /// intensities.
    ///
    /// # Errors
    /// [`Error::Config`] if `rgba` is not `width * height * 4` bytes.
    pub fn upload_rgba(&mut self, rgba: &[u8]) -> Result<()> {
        let expected = (self.width as usize) * (self.height as usize) * 4;
        if rgba.len() != expected {
            return Err(Error::Config(format!(
                "RGBA frame is {} bytes, expected {expected}",
                rgba.len()
            )));
        }
        self.queue.write_buffer(&self.upload_buf, 0, rgba);
        self.dispatch_upload("upload-rgba", false);
        Ok(())
    }

    fn dispatch_upload(&self, label: &str, luma: bool) {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(if luma {
                &self.unpack_luma
            } else {
                &self.rgba_to_luma
            });
            pass.set_bind_group(0, &self.sets[self.current].gray_bind, &[]);
            pass.dispatch_workgroups(self.width.div_ceil(WG_2D), self.height.div_ceil(WG_2D), 1);
        }
        self.queue.submit([enc.finish()]);
    }

    /// Fill levels 1..`levels` of the current set by successive 2x2 box
    /// downsampling.
    ///
    /// # Errors
    /// Never fails today; the signature is fallible because a future
    /// texture-backed path can.
    pub fn build_pyramid(&mut self) -> Result<()> {
        let set = &self.sets[self.current];
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pyramid"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pyramid"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.downsample);
            // Dispatches inside one pass are ordered against each other, so
            // level k reads a level k-1 that is already complete without a
            // submit in between.
            for (k, bind) in set.pyramid_binds.iter().enumerate() {
                let d = self.level_desc[k + 1];
                pass.set_bind_group(0, bind, &[]);
                pass.dispatch_workgroups(d.width.div_ceil(WG_2D), d.height.div_ceil(WG_2D), 1);
            }
        }
        self.queue.submit([enc.finish()]);
        Ok(())
    }

    /// Shi-Tomasi response + non-max suppression. Reads back a few hundred
    /// points, never a full image (spec.md §4 L3).
    ///
    /// `quality` is relative to the strongest response in the frame;
    /// `min_distance` is enforced greedily from the strongest candidate down.
    /// Results are sorted strongest first.
    ///
    /// # Errors
    /// [`Error::Gpu`] if the readback fails.
    pub fn detect_corners(
        &mut self,
        max_corners: usize,
        quality: f32,
        min_distance: f32,
    ) -> Result<Vec<(f32, f32, f32)>> {
        let winners = self.detect_corner_cells()?;
        Ok(reference::select_corners(
            &winners,
            max_corners,
            quality,
            min_distance,
        ))
    }

    /// Run the corner kernels and read back the per-cell winners.
    ///
    /// Split out from [`ImagePipeline::detect_corners`] because this is the
    /// part the GPU actually does, and it is what the equivalence test compares
    /// against `reference::cell_winners`.
    fn detect_corner_cells(&mut self) -> Result<Vec<Option<CellWinner>>> {
        let cells = (self.grid_w * self.grid_h) as u64;
        let set = &self.sets[self.current];
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("corners"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("corners"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &set.corner_bind, &[]);
            pass.set_pipeline(&self.clear_cells);
            pass.dispatch_workgroups((cells as u32).div_ceil(WG_1D), 1, 1);
            let gx = self.width.div_ceil(WG_2D);
            let gy = self.height.div_ceil(WG_2D);
            pass.set_pipeline(&self.shi_tomasi);
            pass.dispatch_workgroups(gx, gy, 1);
            pass.set_pipeline(&self.cell_reduce);
            pass.dispatch_workgroups(gx, gy, 1);
            pass.set_pipeline(&self.cell_pick);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        enc.copy_buffer_to_buffer(&self.cell_max_buf, 0, &self.corner_readback, 0, cells * 4);
        enc.copy_buffer_to_buffer(
            &self.cell_arg_buf,
            0,
            &self.corner_readback,
            cells * 4,
            cells * 4,
        );
        self.queue.submit([enc.finish()]);

        let bytes = read_back(&self.device, &self.corner_readback, cells * 8)?;
        let words: &[u32] = bytemuck::cast_slice(&bytes);
        let (max_bits, arg) = words.split_at(cells as usize);
        Ok(max_bits
            .iter()
            .zip(arg)
            .map(|(&bits, &index)| {
                if index == NO_WINNER {
                    return None;
                }
                Some(CellWinner {
                    x: index % self.width,
                    y: index / self.width,
                    response: f32::from_bits(bits) as f64,
                })
            })
            .collect())
    }

    /// Pyramidal Lucas-Kanade. `prev` must have been uploaded to the other
    /// buffer set via [`ImagePipeline::swap`].
    ///
    /// One workgroup per point; the result list is the same length and order as
    /// `points`.
    ///
    /// # Errors
    /// [`Error::Config`] for an invalid `config` or more than
    /// [`MAX_TRACKED_POINTS`] points, [`Error::Gpu`] if the readback fails.
    pub fn track_flow(
        &mut self,
        points: &[(f32, f32)],
        config: &FlowConfig,
    ) -> Result<Vec<FlowResult>> {
        config.validate()?;
        if points.len() > MAX_TRACKED_POINTS {
            return Err(Error::Config(format!(
                "{} points exceeds MAX_TRACKED_POINTS ({MAX_TRACKED_POINTS})",
                points.len()
            )));
        }
        if points.is_empty() {
            return Ok(Vec::new());
        }

        let flat: Vec<f32> = points.iter().flat_map(|&(x, y)| [x, y]).collect();
        self.queue
            .write_buffer(&self.points_buf, 0, bytemuck::cast_slice(&flat));
        self.queue.write_buffer(
            &self.klt_params_buf,
            0,
            bytemuck::bytes_of(&KltParams {
                levels: self.levels,
                window: config.window,
                iterations: config.iterations,
                point_count: points.len() as u32,
                epsilon: config.epsilon,
                max_error: config.max_error,
                min_eigenvalue: MIN_STRUCTURE_EIGENVALUE as f32,
                border_margin: (config.window / 2) as f32,
            }),
        );

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("flow"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flow"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.track);
            pass.set_bind_group(0, &self.sets[self.current].klt_bind, &[]);
            pass.dispatch_workgroups(points.len() as u32, 1, 1);
        }
        let bytes = (points.len() as u64) * 16;
        enc.copy_buffer_to_buffer(&self.results_buf, 0, &self.flow_readback, 0, bytes);
        self.queue.submit([enc.finish()]);

        let raw = read_back(&self.device, &self.flow_readback, bytes)?;
        let vals: &[f32] = bytemuck::cast_slice(&raw);
        Ok(vals
            .chunks_exact(4)
            .map(|c| FlowResult {
                x: c[0],
                y: c[1],
                ok: c[2] != 0.0,
                error: c[3],
            })
            .collect())
    }

    /// Make the other buffer set current. The set that was current becomes the
    /// template [`ImagePipeline::track_flow`] tracks from.
    pub fn swap(&mut self) {
        self.current ^= 1;
    }

    /// Read the whole Shi-Tomasi response image back.
    ///
    /// Deliberately test-only: a full-image readback is exactly what spec.md §4
    /// L3 forbids on the frame path, and it exists solely so the equivalence
    /// test can compare the kernel against `reference::shi_tomasi_response`
    /// pixel by pixel instead of only through the bucketed summary.
    #[cfg(test)]
    fn debug_read_response(&self) -> Result<Vec<f32>> {
        let bytes = (self.width as u64) * (self.height as u64) * 4;
        let staging = buffer(
            &self.device,
            "response_readback",
            bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&self.response_buf, 0, &staging, 0, bytes);
        self.queue.submit([enc.finish()]);
        let raw = read_back(&self.device, &staging, bytes)?;
        Ok(bytemuck::cast_slice::<u8, f32>(&raw).to_vec())
    }

    /// Read one pyramid level of the current set back as `f32`.
    ///
    /// Test-only for the same reason as [`ImagePipeline::debug_read_response`].
    #[cfg(test)]
    fn debug_read_level(&self, k: u32) -> Result<Vec<f32>> {
        let d = self.level_desc[k as usize];
        let bytes = (d.width as u64) * (d.height as u64) * 4;
        let staging = buffer(
            &self.device,
            "level_readback",
            bytes,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(
            &self.sets[self.current].pyramid,
            (d.offset as u64) * 4,
            &staging,
            0,
            bytes,
        );
        self.queue.submit([enc.finish()]);
        let raw = read_back(&self.device, &staging, bytes)?;
        Ok(bytemuck::cast_slice::<u8, f32>(&raw).to_vec())
    }
}

// ---------------------------------------------------------------------------
// wgpu boilerplate
// ---------------------------------------------------------------------------

fn shader(device: &wgpu::Device, label: &str, src: &str) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    })
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bind_layout(
    device: &wgpu::Device,
    label: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries,
    })
}

fn bind_group(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    resources: &[wgpu::BindingResource<'_>],
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry<'_>> = resources
        .iter()
        .enumerate()
        .map(|(i, r)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: r.clone(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &entries,
    })
}

fn pipeline(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    layout: &wgpu::BindGroupLayout,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn buffer(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

fn uniform<T: Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    value: &[T],
) -> wgpu::Buffer {
    let buf = buffer(
        device,
        label,
        std::mem::size_of_val(value) as u64,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    queue.write_buffer(&buf, 0, bytemuck::cast_slice(value));
    buf
}

/// Map a readback buffer, copy it out and unmap.
///
/// Blocking: `Device::poll` waits for the submission that filled the buffer.
/// This is a host-side wait on GPU completion, not a wall-clock read, so it
/// stays clear of the spec.md §6 no-clock rule.
fn read_back(device: &wgpu::Device, buf: &wgpu::Buffer, size: u64) -> Result<Vec<u8>> {
    let slice = buf.slice(0..size);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r.is_ok());
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| Error::Gpu(format!("device poll failed: {e}")))?;
    match rx.recv() {
        Ok(true) => {}
        Ok(false) => return Err(Error::Gpu("buffer map failed".into())),
        Err(_) => return Err(Error::Gpu("buffer map callback was dropped".into())),
    }
    let out = {
        let view = slice
            .get_mapped_range()
            .map_err(|e| Error::Gpu(format!("mapped range: {e}")))?;
        view.to_vec()
    };
    buf.unmap();
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::FloatImage;
    use crate::testdata::{analytic_texture, checkerboard};

    /// The one device the whole test module shares.
    ///
    /// **Shared, not per-test, and that is load-bearing.** `cargo test` runs
    /// these in parallel threads, and a device per test means N simultaneous
    /// `request_adapter` / `request_device` calls against one GPU. On macOS/Metal
    /// that is merely wasteful; on a Linux box with an NVIDIA card it wedged —
    /// 14 tests sat past 60 s each while the same test alone finished in 0.37 s.
    /// One device removes the contention and the whole suite runs in under a
    /// second.
    ///
    /// A `wgpu::Device` is `Send + Sync` and designed to be shared. Pipelines
    /// are not, so each test still builds its own `ImagePipeline`.
    static SHARED: std::sync::OnceLock<Option<GpuContext>> = std::sync::OnceLock::new();

    /// Borrow the shared device, or `None` when the machine has no adapter.
    ///
    /// spec.md §6 Tier 1 must run on every commit and headless CI frequently has
    /// no GPU, so these tests skip loudly rather than fail.
    fn ctx() -> Option<&'static GpuContext> {
        SHARED
            .get_or_init(|| match GpuContext::new_blocking() {
                Ok(c) => {
                    eprintln!("gpu tests: using {}", c.adapter_name());
                    Some(c)
                }
                Err(e) => {
                    eprintln!("SKIP: no GPU adapter ({e})");
                    None
                }
            })
            .as_ref()
    }

    macro_rules! gpu {
        () => {
            match ctx() {
                Some(c) => c,
                None => return,
            }
        };
    }

    // -- host-side behaviour, no adapter required ---------------------------

    #[test]
    fn flow_config_rejects_even_and_out_of_range_windows() {
        let bad = |w| {
            FlowConfig {
                window: w,
                ..FlowConfig::default()
            }
            .validate()
        };
        assert!(bad(14).is_err(), "even window must be rejected");
        assert!(bad(1).is_err(), "window below 3 must be rejected");
        assert!(bad(MAX_FLOW_WINDOW + 2).is_err());
        assert!(bad(MAX_FLOW_WINDOW).is_ok());
        assert!(FlowConfig {
            iterations: 0,
            ..FlowConfig::default()
        }
        .validate()
        .is_err());
        assert!(FlowConfig {
            max_error: f32::NAN,
            ..FlowConfig::default()
        }
        .validate()
        .is_err());
    }

    #[test]
    fn uniform_blocks_match_their_wgsl_sizes() {
        // A silent mismatch here reads garbage in the shader and produces
        // plausible-looking wrong numbers rather than an error.
        assert_eq!(std::mem::size_of::<GrayParams>(), 16);
        assert_eq!(std::mem::size_of::<PyramidParams>(), 32);
        assert_eq!(std::mem::size_of::<CornerParams>(), 32);
        assert_eq!(std::mem::size_of::<KltParams>(), 32);
        assert_eq!(std::mem::size_of::<LevelDesc>(), 16);
    }

    #[test]
    fn shaders_declare_the_entry_points_the_pipelines_ask_for() {
        for (src, entries) in [
            (GRAYSCALE_WGSL, &["unpack_luma", "rgba_to_luma"][..]),
            (PYRAMID_WGSL, &["downsample_2x2"][..]),
            (
                CORNERS_WGSL,
                &["clear_cells", "shi_tomasi", "cell_reduce", "cell_pick"][..],
            ),
            (KLT_WGSL, &["track"][..]),
        ] {
            for e in entries {
                assert!(src.contains(&format!("fn {e}(")), "missing entry point {e}");
            }
        }
    }

    #[test]
    fn corner_radius_agrees_between_shader_and_reference() {
        assert!(CORNERS_WGSL.contains(&format!(
            "const RADIUS: i32 = {};",
            reference::CORNER_RADIUS
        )));
        assert!(KLT_WGSL.contains(&format!("const WG: u32 = {}u;", reference::KLT_WORKGROUP)));
    }

    // -- GPU vs reference ----------------------------------------------------

    #[test]
    fn gpu_level0_matches_the_uploaded_image_exactly() {
        let ctx = gpu!();
        let img = analytic_texture(64, 48, 0.0, 0.0);
        let mut pipe = ImagePipeline::new(ctx, 64, 48, 3).unwrap();
        pipe.upload(&img).unwrap();
        let level0 = pipe.debug_read_level(0).unwrap();
        for (i, &v) in level0.iter().enumerate() {
            // u8 widened to f32 is exact; anything else is an indexing bug.
            assert_eq!(v, img.data()[i] as f32, "pixel {i}");
        }
    }

    #[test]
    fn gpu_rgba_conversion_matches_core_weights() {
        let ctx = gpu!();
        let (w, h) = (32u32, 16u32);
        let rgba: Vec<u8> = (0..(w * h * 4)).map(|i| (i * 37 % 256) as u8).collect();
        let expect = GrayImage::from_rgba(w, h, &rgba);
        let mut pipe = ImagePipeline::new(ctx, w, h, 1).unwrap();
        pipe.upload_rgba(&rgba).unwrap();
        let level0 = pipe.debug_read_level(0).unwrap();
        for (i, &v) in level0.iter().enumerate() {
            assert_eq!(v, expect.data()[i] as f32, "pixel {i}");
        }
    }

    #[test]
    fn gpu_pyramid_matches_reference_within_f32_rounding() {
        let ctx = gpu!();
        let (w, h) = (100u32, 70u32); // odd-ish, so the clamped edges get exercised
        let img = analytic_texture(w, h, 0.0, 0.0);
        let expect = reference::build_pyramid(&img, 4);

        let mut pipe = ImagePipeline::new(ctx, w, h, 4).unwrap();
        pipe.upload(&img).unwrap();
        pipe.build_pyramid().unwrap();

        for k in 0..4u32 {
            let got = pipe.debug_read_level(k).unwrap();
            let want = &expect[k as usize];
            assert_eq!(
                (want.width(), want.height()),
                pipe.level_size(k).unwrap(),
                "level {k} dimensions"
            );
            assert_eq!(got.len(), want.data().len());
            for (i, (&g, &r)) in got.iter().zip(want.data()).enumerate() {
                // Intensities live in 0..255 and the box filter is exact in
                // f32 for the first few levels; 1e-3 is generous.
                assert!(
                    (g as f64 - r).abs() < 1e-3,
                    "level {k} pixel {i}: gpu {g} vs reference {r}"
                );
            }
        }
    }

    #[test]
    fn gpu_shi_tomasi_matches_reference_pixel_for_pixel() {
        let ctx = gpu!();
        let (w, h) = (96u32, 64u32);
        let img = analytic_texture(w, h, 0.0, 0.0);
        let want = reference::shi_tomasi_response(&FloatImage::from_gray(&img));

        let mut pipe = ImagePipeline::new(ctx, w, h, 1).unwrap();
        pipe.upload(&img).unwrap();
        pipe.build_pyramid().unwrap();
        let _ = pipe.detect_corners(500, 0.0, 0.0).unwrap();
        let got = pipe.debug_read_response().unwrap();

        let peak = want.iter().copied().fold(0.0f64, f64::max);
        assert!(peak > 1.0, "fixture has no corner energy");
        for (i, (&g, &r)) in got.iter().zip(&want).enumerate() {
            // Relative to the peak response: the tensor entries are sums of
            // squared gradients, so absolute magnitudes are in the thousands.
            assert!(
                (g as f64 - r).abs() < 1e-4 * peak,
                "pixel {i}: gpu {g} vs reference {r} (peak {peak})"
            );
        }
    }

    #[test]
    fn gpu_cell_winners_match_reference() {
        let ctx = gpu!();
        let (w, h) = (128u32, 96u32);
        let img = analytic_texture(w, h, 0.0, 0.0);
        let response = reference::shi_tomasi_response(&FloatImage::from_gray(&img));
        let want = reference::cell_winners(&response, w, h, CORNER_CELL_PX);

        let mut pipe = ImagePipeline::new(ctx, w, h, 1).unwrap();
        pipe.upload(&img).unwrap();
        pipe.build_pyramid().unwrap();
        let got = pipe.detect_corner_cells().unwrap();

        assert_eq!(got.len(), want.len());
        for (i, (g, r)) in got.iter().zip(&want).enumerate() {
            match (g, r) {
                (None, None) => {}
                (Some(g), Some(r)) => {
                    assert_eq!((g.x, g.y), (r.x, r.y), "cell {i} picked a different pixel");
                    assert!(
                        (g.response - r.response).abs() < 1e-3 * r.response.max(1.0),
                        "cell {i}: gpu {} vs reference {}",
                        g.response,
                        r.response
                    );
                }
                _ => panic!("cell {i}: gpu {g:?} vs reference {r:?}"),
            }
        }
    }

    #[test]
    fn gpu_corners_land_on_checkerboard_intersections() {
        let ctx = gpu!();
        const S: u32 = 20;
        const N: u32 = 160;
        let img = checkerboard(N, N, S);
        let mut pipe = ImagePipeline::new(ctx, N, N, 1).unwrap();
        pipe.upload(&img).unwrap();
        pipe.build_pyramid().unwrap();
        let corners = pipe.detect_corners(256, 0.05, 8.0).unwrap();

        let expected = reference::detect_corners(&FloatImage::from_gray(&img), 256, 0.05, 8.0);
        assert_eq!(corners.len(), expected.len(), "gpu {corners:?}");
        assert_eq!(corners.len(), 49);
        for (&(gx, gy, _), &(rx, ry, _)) in corners.iter().zip(&expected) {
            assert_eq!((gx, gy), (rx, ry));
            // And the answer we knew before running anything: every corner sits
            // on a block intersection.
            assert!(
                (gx as u32).is_multiple_of(S) || (gx as u32 + 1).is_multiple_of(S),
                "x = {gx} is not on a block boundary"
            );
            assert!((gy as u32).is_multiple_of(S) || (gy as u32 + 1).is_multiple_of(S));
        }
    }

    #[test]
    fn gpu_flow_recovers_a_known_subpixel_translation() {
        let ctx = gpu!();
        const DX: f64 = 0.37;
        const DY: f64 = -0.62;
        let (w, h) = (128u32, 96u32);
        let a = analytic_texture(w, h, 0.0, 0.0);
        let b = analytic_texture(w, h, DX, DY);

        let mut pipe = ImagePipeline::new(ctx, w, h, 3).unwrap();
        pipe.upload(&a).unwrap();
        pipe.build_pyramid().unwrap();
        pipe.swap();
        pipe.upload(&b).unwrap();
        pipe.build_pyramid().unwrap();

        let points = [(40.0f32, 44.0f32), (60.0, 52.0), (75.0, 66.0)];
        let out = pipe.track_flow(&points, &FlowConfig::default()).unwrap();
        for (r, p) in out.iter().zip(&points) {
            assert!(r.ok, "{p:?} lost: {r:?}");
            assert!(
                (r.x as f64 - (p.0 as f64 + DX)).abs() < 0.05,
                "x error {:.4} at {p:?}",
                r.x as f64 - (p.0 as f64 + DX)
            );
            assert!(
                (r.y as f64 - (p.1 as f64 + DY)).abs() < 0.05,
                "y error {:.4} at {p:?}",
                r.y as f64 - (p.1 as f64 + DY)
            );
        }
    }

    #[test]
    fn gpu_flow_matches_reference() {
        let ctx = gpu!();
        let (w, h) = (128u32, 96u32);
        let a = analytic_texture(w, h, 0.0, 0.0);
        let b = analytic_texture(w, h, 1.85, -2.4);

        let mut pipe = ImagePipeline::new(ctx, w, h, 3).unwrap();
        pipe.upload(&a).unwrap();
        pipe.build_pyramid().unwrap();
        pipe.swap();
        pipe.upload(&b).unwrap();
        pipe.build_pyramid().unwrap();

        let points: Vec<(f32, f32)> = (0..24)
            .map(|i| (30.0 + (i % 6) as f32 * 11.0, 30.0 + (i / 6) as f32 * 13.0))
            .collect();
        let cfg = FlowConfig::default();
        let got = pipe.track_flow(&points, &cfg).unwrap();
        let want = reference::track_flow(
            &reference::build_pyramid(&a, 3),
            &reference::build_pyramid(&b, 3),
            &points,
            &cfg,
        );

        assert_eq!(got.len(), want.len());
        for (i, (g, r)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.ok, r.ok, "point {i} disagrees on validity: {g:?} {r:?}");
            if r.ok {
                // f32 on the GPU against f64 in the reference, with the same
                // summation order: 0.01 px is a hundred times the divergence we
                // actually see and still far below anything that would matter.
                assert!(
                    (g.x - r.x).abs() < 0.01 && (g.y - r.y).abs() < 0.01,
                    "point {i}: gpu ({}, {}) vs reference ({}, {})",
                    g.x,
                    g.y,
                    r.x,
                    r.y
                );
                assert!((g.error - r.error).abs() < 0.05, "point {i} error");
            }
        }
    }

    #[test]
    fn gpu_flow_matches_reference_at_every_window_size() {
        // The lane loop is `for k = tid; k < window^2; k += 64`. At window 3
        // only nine of the sixty-four lanes contribute anything and the rest
        // must reduce as exact zeros; at window 31 each lane walks fifteen
        // pixels. Both ends have to agree with the reference.
        let ctx = gpu!();
        let (w, h) = (128u32, 128u32);
        let a = analytic_texture(w, h, 0.0, 0.0);
        let b = analytic_texture(w, h, 0.6, -0.45);
        let mut pipe = ImagePipeline::new(ctx, w, h, 3).unwrap();
        pipe.upload(&a).unwrap();
        pipe.build_pyramid().unwrap();
        pipe.swap();
        pipe.upload(&b).unwrap();
        pipe.build_pyramid().unwrap();

        let ref_prev = reference::build_pyramid(&a, 3);
        let ref_next = reference::build_pyramid(&b, 3);
        let points = [(56.0f32, 60.0f32), (72.0, 48.0)];
        for &window in &[3u32, 7, 21, MAX_FLOW_WINDOW] {
            let cfg = FlowConfig {
                window,
                ..FlowConfig::default()
            };
            let got = pipe.track_flow(&points, &cfg).unwrap();
            let want = reference::track_flow(&ref_prev, &ref_next, &points, &cfg);
            for (i, (g, r)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.ok, r.ok, "window {window} point {i}: {g:?} {r:?}");
                if r.ok {
                    assert!(
                        (g.x - r.x).abs() < 0.01 && (g.y - r.y).abs() < 0.01,
                        "window {window} point {i}: gpu {g:?} reference {r:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn gpu_flow_rejects_a_flat_patch() {
        let ctx = gpu!();
        let flat = GrayImage::from_vec(64, 64, vec![100; 64 * 64]);
        let mut pipe = ImagePipeline::new(ctx, 64, 64, 3).unwrap();
        pipe.upload(&flat).unwrap();
        pipe.build_pyramid().unwrap();
        pipe.swap();
        pipe.upload(&flat).unwrap();
        pipe.build_pyramid().unwrap();
        let out = pipe
            .track_flow(&[(32.0, 32.0)], &FlowConfig::default())
            .unwrap();
        assert!(!out[0].ok, "a singular system must not report a track");
    }

    #[test]
    fn gpu_flow_is_reproducible_across_runs() {
        // spec.md §6: the same binary must replay bit-for-bit. Two identical
        // dispatches must therefore agree exactly, not approximately.
        let ctx = gpu!();
        let (w, h) = (96u32, 96u32);
        let a = analytic_texture(w, h, 0.0, 0.0);
        let b = analytic_texture(w, h, 0.9, 1.3);
        let mut pipe = ImagePipeline::new(ctx, w, h, 3).unwrap();
        pipe.upload(&a).unwrap();
        pipe.build_pyramid().unwrap();
        pipe.swap();
        pipe.upload(&b).unwrap();
        pipe.build_pyramid().unwrap();
        let points: Vec<(f32, f32)> = (0..16)
            .map(|i| (24.0 + (i % 4) as f32 * 12.0, 24.0 + (i / 4) as f32 * 12.0))
            .collect();
        let cfg = FlowConfig::default();
        let first = pipe.track_flow(&points, &cfg).unwrap();
        let second = pipe.track_flow(&points, &cfg).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn swap_selects_the_other_template() {
        // Tracking a frame against itself must give zero flow; tracking it
        // against a shifted frame must not. If swap() bound the wrong set both
        // would look the same.
        let ctx = gpu!();
        let (w, h) = (96u32, 96u32);
        let a = analytic_texture(w, h, 0.0, 0.0);
        let b = analytic_texture(w, h, 3.0, 0.0);
        let mut pipe = ImagePipeline::new(ctx, w, h, 3).unwrap();
        let p = [(48.0f32, 48.0f32)];
        let cfg = FlowConfig::default();

        pipe.upload(&a).unwrap();
        pipe.build_pyramid().unwrap();
        pipe.swap();
        pipe.upload(&a).unwrap();
        pipe.build_pyramid().unwrap();
        let still = pipe.track_flow(&p, &cfg).unwrap();
        assert!((still[0].x - 48.0).abs() < 0.01, "{still:?}");

        pipe.swap();
        pipe.upload(&b).unwrap();
        pipe.build_pyramid().unwrap();
        let moved = pipe.track_flow(&p, &cfg).unwrap();
        assert!((moved[0].x - 51.0).abs() < 0.05, "{moved:?}");
    }

    #[test]
    fn rejects_a_frame_of_the_wrong_size() {
        let ctx = gpu!();
        let mut pipe = ImagePipeline::new(ctx, 64, 64, 2).unwrap();
        assert!(pipe.upload(&GrayImage::new(64, 32)).is_err());
        assert!(pipe.upload_rgba(&[0u8; 10]).is_err());
    }

    #[test]
    fn rejects_degenerate_construction() {
        let ctx = gpu!();
        assert!(ImagePipeline::new(ctx, 0, 32, 2).is_err());
        assert!(ImagePipeline::new(ctx, 32, 32, 0).is_err());
    }

    #[test]
    fn rejects_more_points_than_capacity() {
        let ctx = gpu!();
        let mut pipe = ImagePipeline::new(ctx, 64, 64, 2).unwrap();
        let too_many = vec![(1.0f32, 1.0f32); MAX_TRACKED_POINTS + 1];
        assert!(pipe.track_flow(&too_many, &FlowConfig::default()).is_err());
        assert!(pipe
            .track_flow(&[], &FlowConfig::default())
            .unwrap()
            .is_empty());
    }

    #[test]
    fn limits_are_at_least_the_webgpu_baseline() {
        let ctx = gpu!();
        let l = ctx.limits();
        assert!(l.max_workgroup_size >= 256, "{l:?}");
        assert!(l.max_buffer_bytes >= 256 << 20, "{l:?}");
        assert!(!ctx.adapter_name().is_empty());
    }
}
