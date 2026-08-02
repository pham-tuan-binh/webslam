// 2x2 box downsample, one pyramid level per dispatch.
//
// Source and destination live in the same buffer at disjoint offsets. WGSL
// cannot bind one buffer twice with different access modes, and a runtime array
// of per-level bindings does not exist, so the whole pyramid is one allocation
// and the level table is passed in the uniform.
//
// Level k has dimensions `max(w >> k, 1) x max(h >> k, 1)`. Destination pixel
// (x, y) averages source pixels (2x, 2y), (2x+1, 2y), (2x, 2y+1), (2x+1, 2y+1)
// with border clamping, which places the destination pixel centre at source
// coordinate 2x + 0.5 — the half-pixel convention
// `wslam_core::CameraIntrinsics::scaled` already uses.

struct Params {
    src_offset: u32,
    src_w: u32,
    src_h: u32,
    dst_offset: u32,
    dst_w: u32,
    dst_h: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> pyr: array<f32>;

fn fetch_src(x: i32, y: i32) -> f32 {
    let cx = u32(clamp(x, 0, i32(params.src_w) - 1));
    let cy = u32(clamp(y, 0, i32(params.src_h) - 1));
    return pyr[params.src_offset + cy * params.src_w + cx];
}

@compute @workgroup_size(8, 8)
fn downsample_2x2(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.dst_w || gid.y >= params.dst_h {
        return;
    }
    let sx = i32(gid.x) * 2;
    let sy = i32(gid.y) * 2;
    // Summation order is fixed and mirrored by `reference::downsample_2x2`;
    // f32 addition is not associative and the CPU reference is the oracle.
    let s = ((fetch_src(sx, sy) + fetch_src(sx + 1, sy)) + fetch_src(sx, sy + 1))
        + fetch_src(sx + 1, sy + 1);
    pyr[params.dst_offset + gid.y * params.dst_w + gid.x] = s * 0.25;
}
