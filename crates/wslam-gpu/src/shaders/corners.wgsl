// Shi-Tomasi corner response + grid-bucketed non-max suppression.
//
// spec.md §4 L3: "read back a few hundred points, never a full image". The
// response image never leaves the GPU. What comes back is one candidate per
// grid cell — for a 640x480 frame with 16 px cells that is 1200 u32 pairs, 9.6
// kB, against 300 kB for the image itself.
//
// Bucketing is what makes the readback bounded *and* spreads the features over
// the frame, which is what the tracker actually wants: a hundred corners piled
// onto one high-contrast object is a degenerate PnP configuration.
//
// Four dispatches in one pass; WebGPU orders dispatches within a pass, so the
// three global barriers this needs are free.

struct Params {
    // Element offset of the level this runs on inside the pyramid buffer.
    offset: u32,
    width: u32,
    height: u32,
    cell_px: u32,
    grid_w: u32,
    grid_h: u32,
    cell_count: u32,
    _pad: u32,
}

// Structure-tensor box radius. Fixed rather than a uniform so the loop is
// statically bounded; `reference::CORNER_RADIUS` must match.
const RADIUS: i32 = 1;
// Central differences need one pixel outside the box, so the response is only
// defined RADIUS + 1 pixels in from the border.
const MARGIN: i32 = RADIUS + 1;

const NO_WINNER: u32 = 0xffffffffu;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> pyr: array<f32>;
@group(0) @binding(2) var<storage, read_write> response: array<f32>;
@group(0) @binding(3) var<storage, read_write> cell_max: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> cell_arg: array<atomic<u32>>;

fn fetch(x: i32, y: i32) -> f32 {
    let cx = u32(clamp(x, 0, i32(params.width) - 1));
    let cy = u32(clamp(y, 0, i32(params.height) - 1));
    return pyr[params.offset + cy * params.width + cx];
}

@compute @workgroup_size(64)
fn clear_cells(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.cell_count {
        return;
    }
    atomicStore(&cell_max[gid.x], 0u);
    atomicStore(&cell_arg[gid.x], NO_WINNER);
}

@compute @workgroup_size(8, 8)
fn shi_tomasi(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    var r = 0.0;

    let inside = x >= MARGIN && y >= MARGIN
        && x < i32(params.width) - MARGIN && y < i32(params.height) - MARGIN;
    if inside {
        var a = 0.0;
        var b = 0.0;
        var c = 0.0;
        // Row-major traversal; `reference::shi_tomasi_response` accumulates in
        // the same order.
        for (var dy = -RADIUS; dy <= RADIUS; dy = dy + 1) {
            for (var dx = -RADIUS; dx <= RADIUS; dx = dx + 1) {
                let ix = 0.5 * (fetch(x + dx + 1, y + dy) - fetch(x + dx - 1, y + dy));
                let iy = 0.5 * (fetch(x + dx, y + dy + 1) - fetch(x + dx, y + dy - 1));
                a = a + ix * ix;
                b = b + ix * iy;
                c = c + iy * iy;
            }
        }
        // Smaller eigenvalue of [[a, b], [b, c]] (Shi & Tomasi 1994). The
        // half-trace form is used rather than the quadratic formula because
        // a + c can be large and the discriminant small.
        let half_trace = 0.5 * (a + c);
        let half_diff = 0.5 * (a - c);
        r = half_trace - sqrt(half_diff * half_diff + b * b);
        // A PSD tensor cannot have a negative eigenvalue; only rounding can
        // produce one, and a negative response would break the u32 ordering
        // that `cell_reduce` relies on.
        r = max(r, 0.0);
    }
    response[y * i32(params.width) + x] = r;
}

@compute @workgroup_size(8, 8)
fn cell_reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height {
        return;
    }
    let r = response[gid.y * params.width + gid.x];
    if r <= 0.0 {
        return;
    }
    let cell = (gid.y / params.cell_px) * params.grid_w + (gid.x / params.cell_px);
    // IEEE-754 non-negative floats order identically to their bit patterns read
    // as u32, so an integer atomicMax is an exact float max. WGSL has no f32
    // atomics.
    atomicMax(&cell_max[cell], bitcast<u32>(r));
}

@compute @workgroup_size(8, 8)
fn cell_pick(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height {
        return;
    }
    let index = gid.y * params.width + gid.x;
    let r = response[index];
    if r <= 0.0 {
        return;
    }
    let cell = (gid.y / params.cell_px) * params.grid_w + (gid.x / params.cell_px);
    if bitcast<u32>(r) == atomicLoad(&cell_max[cell]) {
        // Ties resolved to the lowest raster index. Without this the winner
        // would depend on dispatch order and the pipeline would stop being
        // reproducible, which spec.md §6 makes non-negotiable.
        atomicMin(&cell_arg[cell], index);
    }
}
