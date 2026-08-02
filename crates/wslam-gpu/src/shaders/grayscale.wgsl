// Upload conversion: host bytes -> f32 luma at pyramid level 0.
//
// spec.md §4 L3 puts the whole image front-end on the GPU. The browser hands us
// RGBA from a canvas or an external texture, so the colour conversion belongs
// here rather than on the CPU where it would cost a full-image pass before the
// upload it is meant to feed.
//
// Both entry points write into the *shared* pyramid buffer at `dst_offset`, so
// no copy is needed between this kernel and pyramid.wgsl.

struct Params {
    width: u32,
    height: u32,
    // Element offset of level 0 inside the shared pyramid buffer.
    dst_offset: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// Host bytes reinterpreted as words. Little-endian: WebGPU buffer contents are
// host-byte-ordered and every implementation of it is little-endian, which is
// also what `u32::from_le_bytes` on the Rust side assumes.
@group(0) @binding(1) var<storage, read> packed_src: array<u32>;
@group(0) @binding(2) var<storage, read_write> pyramid: array<f32>;

// One 8-bit luma sample per byte, four per word.
@compute @workgroup_size(8, 8)
fn unpack_luma(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height {
        return;
    }
    let i = gid.y * params.width + gid.x;
    let word = packed_src[i >> 2u];
    let byte = (word >> ((i & 3u) * 8u)) & 0xffu;
    pyramid[params.dst_offset + i] = f32(byte);
}

// One RGBA8 pixel per word.
@compute @workgroup_size(8, 8)
fn rgba_to_luma(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height {
        return;
    }
    let i = gid.y * params.width + gid.x;
    let px = packed_src[i];
    let r = px & 0xffu;
    let g = (px >> 8u) & 0xffu;
    let b = (px >> 16u) & 0xffu;
    // Rec. 601 weights as integers scaled by 2^16, matching
    // `wslam_core::GrayImage::from_rgba` exactly. Float weights would make the
    // native and browser paths disagree in the last bit, and spec.md §6 L3
    // requires that any divergence be attributable to a port bug rather than to
    // rounding we chose not to pin down.
    let y = (19595u * r + 38470u * g + 7471u * b) >> 16u;
    pyramid[params.dst_offset + i] = f32(y);
}
