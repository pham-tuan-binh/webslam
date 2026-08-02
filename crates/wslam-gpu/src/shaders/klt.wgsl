// Pyramidal Lucas-Kanade, one workgroup per tracked point.
//
// One workgroup per point rather than one invocation per point: the window sum
// is the whole cost of the kernel, and a 15x15 window is 225 samples that a
// single invocation would walk serially while 63 lanes idle. Splitting the
// window across the workgroup turns it into a strided load plus a tree
// reduction.
//
// The formulation is Bouguet's: the structure tensor and the residual gradient
// both come from the *template* (the previous frame), so `G` is the same matrix
// Shi-Tomasi scores and the per-iteration work is one warp of samples from the
// current frame. Point at level L is `(p + 0.5) / 2^L - 0.5`, the half-pixel
// convention `wslam_core::CameraIntrinsics::scaled` uses, so a displacement
// scales by exactly 2 between levels.
//
// Determinism (spec.md §6): the reduction is a fixed strided partition followed
// by a fixed binary tree, and `reference::track_flow` sums in exactly that
// order. Float addition is not associative, so "same algorithm" is not enough —
// the summation order has to be part of the contract.

struct Params {
    levels: u32,
    window: u32,
    iterations: u32,
    point_count: u32,
    epsilon: f32,
    max_error: f32,
    // Reject the point if lambda_min / window_area falls below this. Scale-free
    // so it does not have to be retuned with the window size.
    min_eigenvalue: f32,
    border_margin: f32,
}

struct Level {
    offset: u32,
    width: u32,
    height: u32,
    _pad: u32,
}

const WG: u32 = 64u;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> prev_pyr: array<f32>;
@group(0) @binding(2) var<storage, read> next_pyr: array<f32>;
@group(0) @binding(3) var<storage, read> levels: array<Level>;
@group(0) @binding(4) var<storage, read> points: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> results: array<vec4<f32>>;

// (Ixx, Ixy, Iyy) and (bx, by, |residual|) partial sums.
var<workgroup> acc_g: array<vec3<f32>, 64>;
var<workgroup> acc_b: array<vec3<f32>, 64>;

// Border-clamped fetch, matching `wslam_core::GrayImage::at`. Two copies rather
// than one function over a pointer: passing a storage pointer as a parameter
// needs the `unrestricted_pointer_parameters` language extension, which is not
// universally available.
fn fetch_prev(off: u32, w: u32, h: u32, x: i32, y: i32) -> f32 {
    let cx = u32(clamp(x, 0, i32(w) - 1));
    let cy = u32(clamp(y, 0, i32(h) - 1));
    return prev_pyr[off + cy * w + cx];
}

fn fetch_next(off: u32, w: u32, h: u32, x: i32, y: i32) -> f32 {
    let cx = u32(clamp(x, 0, i32(w) - 1));
    let cy = u32(clamp(y, 0, i32(h) - 1));
    return next_pyr[off + cy * w + cx];
}

fn sample_prev(off: u32, w: u32, h: u32, x: f32, y: f32) -> f32 {
    let bx = floor(x);
    let by = floor(y);
    let fx = x - bx;
    let fy = y - by;
    let x0 = i32(bx);
    let y0 = i32(by);
    let p00 = fetch_prev(off, w, h, x0, y0);
    let p10 = fetch_prev(off, w, h, x0 + 1, y0);
    let p01 = fetch_prev(off, w, h, x0, y0 + 1);
    let p11 = fetch_prev(off, w, h, x0 + 1, y0 + 1);
    let top = p00 + (p10 - p00) * fx;
    let bot = p01 + (p11 - p01) * fx;
    return top + (bot - top) * fy;
}

fn sample_next(off: u32, w: u32, h: u32, x: f32, y: f32) -> f32 {
    let bx = floor(x);
    let by = floor(y);
    let fx = x - bx;
    let fy = y - by;
    let x0 = i32(bx);
    let y0 = i32(by);
    let p00 = fetch_next(off, w, h, x0, y0);
    let p10 = fetch_next(off, w, h, x0 + 1, y0);
    let p01 = fetch_next(off, w, h, x0, y0 + 1);
    let p11 = fetch_next(off, w, h, x0 + 1, y0 + 1);
    let top = p00 + (p10 - p00) * fx;
    let bot = p01 + (p11 - p01) * fx;
    return top + (bot - top) * fy;
}

@compute @workgroup_size(64)
fn track(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let pid = wid.x;
    let tid = lid.x;
    if pid >= params.point_count {
        return;
    }

    let p0 = points[pid];
    let half = i32(params.window / 2u);
    let wcount = params.window * params.window;
    let inv_n = 1.0 / f32(wcount);
    // One pass beyond the solver iterations, so the reported error is measured
    // at the flow we actually return rather than one Gauss-Newton step behind.
    let passes = params.iterations + 1u;

    var g = vec2<f32>(0.0, 0.0);
    var failed = false;
    var err = 0.0;

    for (var li = i32(params.levels) - 1; li >= 0; li = li - 1) {
        let lv = levels[u32(li)];
        let s = 1.0 / f32(1u << u32(li));
        let p = (p0 + vec2<f32>(0.5, 0.5)) * s - vec2<f32>(0.5, 0.5);
        var converged = false;

        for (var it = 0u; it < passes; it = it + 1u) {
            var a = 0.0;
            var b = 0.0;
            var c = 0.0;
            var rx = 0.0;
            var ry = 0.0;
            var e = 0.0;
            for (var k = tid; k < wcount; k = k + WG) {
                let dx = f32(i32(k % params.window) - half);
                let dy = f32(i32(k / params.window) - half);
                let tx = p.x + dx;
                let ty = p.y + dy;
                let tmpl = sample_prev(lv.offset, lv.width, lv.height, tx, ty);
                let ix = 0.5 * (
                    sample_prev(lv.offset, lv.width, lv.height, tx + 1.0, ty)
                    - sample_prev(lv.offset, lv.width, lv.height, tx - 1.0, ty)
                );
                let iy = 0.5 * (
                    sample_prev(lv.offset, lv.width, lv.height, tx, ty + 1.0)
                    - sample_prev(lv.offset, lv.width, lv.height, tx, ty - 1.0)
                );
                let warped = sample_next(lv.offset, lv.width, lv.height, tx + g.x, ty + g.y);
                let d = tmpl - warped;
                a = a + ix * ix;
                b = b + ix * iy;
                c = c + iy * iy;
                rx = rx + d * ix;
                ry = ry + d * iy;
                e = e + abs(d);
            }

            acc_g[tid] = vec3<f32>(a, b, c);
            acc_b[tid] = vec3<f32>(rx, ry, e);
            workgroupBarrier();
            for (var stride = WG / 2u; stride > 0u; stride = stride >> 1u) {
                if tid < stride {
                    acc_g[tid] = acc_g[tid] + acc_g[tid + stride];
                    acc_b[tid] = acc_b[tid] + acc_b[tid + stride];
                }
                workgroupBarrier();
            }
            let gsum = acc_g[0];
            let bsum = acc_b[0];
            // Every lane has read the reduction result; the next pass will
            // overwrite slot 0.
            workgroupBarrier();

            err = bsum.z * inv_n;

            // Uniform across the workgroup: every lane sees the same reduction,
            // so every lane keeps the same `g`. The update is deliberately not
            // a `break`, because WGSL's uniformity analysis treats a value read
            // from workgroup memory as non-uniform and would then reject the
            // barriers above.
            let det = gsum.x * gsum.z - gsum.y * gsum.y;
            let half_trace = 0.5 * (gsum.x + gsum.z);
            let half_diff = 0.5 * (gsum.x - gsum.z);
            let lmin = half_trace - sqrt(half_diff * half_diff + gsum.y * gsum.y);
            if it < params.iterations && !failed && !converged {
                if det <= 0.0 || lmin < params.min_eigenvalue * f32(wcount) {
                    failed = true;
                } else {
                    let nu = vec2<f32>(
                        (gsum.z * bsum.x - gsum.y * bsum.y) / det,
                        (gsum.x * bsum.y - gsum.y * bsum.x) / det,
                    );
                    g = g + nu;
                    if sqrt(nu.x * nu.x + nu.y * nu.y) < params.epsilon {
                        converged = true;
                    }
                }
            }
        }

        if li > 0 {
            g = g * 2.0;
        }
    }

    let base = levels[0];
    let m = params.border_margin;
    let tracked = p0 + g;
    // Negated conjunction, so a NaN coordinate fails the test instead of
    // sliding through every individual comparison.
    var ok = !failed;
    if !(tracked.x >= m && tracked.y >= m
        && tracked.x <= f32(base.width - 1u) - m
        && tracked.y <= f32(base.height - 1u) - m) {
        ok = false;
    }
    if !(err <= params.max_error) {
        ok = false;
    }

    if tid == 0u {
        results[pid] = vec4<f32>(tracked.x, tracked.y, select(0.0, 1.0, ok), err);
    }
}
