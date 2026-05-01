// [wllama-fork 2026-05-01] Phase 3 Fusion #1: RMS_NORM + MUL + ADD
//
// Fuses the gemma-3n / Llama-style "post-block residual" pattern:
//   tmp_a = RMS_NORM(rn_src)
//   tmp_b = MUL(tmp_a, weight)
//   dst   = ADD(tmp_b, residual)
//
// Math: dst[i,j] = (1/sqrt(mean(rn_src[j,*]^2)+eps)) * rn_src[i,j] * w[i_w,j] + residual[i,j]
//
// 72 fires per decode token in gemma-3n E2B (post-attn O+norm+residual,
// post-FFN down+norm+residual; both per-layer, ×36 layers).
//
// Two variants needed because ggml's allocator aliases buffers across
// fused chains:
//   NORMAL:    rn_src, mul_w, residual, dst all distinct buffers
//   INPLACE_RN: dst aliases rn_src (the common gemma-3n case — rn_src is
//               dead after the chain so allocator reuses its buffer)
//
// WebGPU rule: a buffer cannot be bound as both `read` and `read_write`
// in the same compute-pass synchronization scope, EVEN IF declared with
// different access modes. So when aliasing happens we must collapse the
// two bindings into ONE (read_write).

#ifdef INPLACE_RN_DST
// Variant: dst aliases rn_src. They share binding 0 (read_write).
@group(0) @binding(0)
var<storage, read_write> rn_src_dst: array<f32>;

@group(0) @binding(1)
var<storage, read_write> mul_w: array<f32>;

@group(0) @binding(2)
var<storage, read_write> residual: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

fn read_rn(off: u32) -> f32 { return rn_src_dst[off]; }
fn write_dst(off: u32, v: f32) { rn_src_dst[off] = v; }

#else
// Variant: all 4 buffers distinct. rn_src/mul_w/residual are read-only,
// dst is read_write.
@group(0) @binding(0)
var<storage, read_write> rn_src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> mul_w: array<f32>;

@group(0) @binding(2)
var<storage, read_write> residual: array<f32>;

@group(0) @binding(3)
var<storage, read_write> dst: array<f32>;

@group(0) @binding(4)
var<uniform> params: Params;

fn read_rn(off: u32) -> f32 { return rn_src[off]; }
fn write_dst(off: u32, v: f32) { dst[off] = v; }
#endif

struct Params {
    offset_rn:   u32,
    offset_w:    u32,
    offset_res:  u32,
    offset_dst:  u32,

    stride_rn1:  u32,
    stride_rn2:  u32,
    stride_rn3:  u32,

    stride_w1:   u32,
    stride_w2:   u32,
    stride_w3:   u32,

    stride_res1: u32,
    stride_res2: u32,
    stride_res3: u32,

    stride_d1:   u32,
    stride_d2:   u32,
    stride_d3:   u32,

    ne0:         u32,
    ne1:         u32,
    ne2:         u32,
    ne3:         u32,

    w_ne0:       u32,
    w_ne1:       u32,
    w_ne2:       u32,
    w_ne3:       u32,

    eps:         f32,
};

var<workgroup> scratch: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id)        wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {

    var i = wid.x;
    let i3 = i / (params.ne2 * params.ne1);
    i      = i % (params.ne2 * params.ne1);
    let i2 = i / params.ne1;
    let i1 = i % params.ne1;

    let row_rn  = params.offset_rn  + i3 * params.stride_rn3  + i2 * params.stride_rn2  + i1 * params.stride_rn1;
    let row_w   = params.offset_w
                  + (i3 % params.w_ne3) * params.stride_w3
                  + (i2 % params.w_ne2) * params.stride_w2
                  + (i1 % params.w_ne1) * params.stride_w1;
    let row_res = params.offset_res + i3 * params.stride_res3 + i2 * params.stride_res2 + i1 * params.stride_res1;
    let row_d   = params.offset_dst + i3 * params.stride_d3   + i2 * params.stride_d2   + i1 * params.stride_d1;

    // Phase 1: per-thread sum-of-squares of rn_src.
    let elems_per_thread = (params.ne0 + WG_SIZE - 1u) / WG_SIZE;
    var sum: f32 = 0.0;
    var col = lid.x;
    for (var k: u32 = 0u; k < elems_per_thread; k = k + 1u) {
        if (col >= params.ne0) {
            break;
        }
        let v = read_rn(row_rn + col);
        sum = sum + v * v;
        col = col + WG_SIZE;
    }
    scratch[lid.x] = sum;
    workgroupBarrier();

    // Phase 2: tree reduction.
    var offset: u32 = WG_SIZE / 2u;
    while (offset > 0u) {
        if (lid.x < offset) {
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + offset];
        }
        offset = offset / 2u;
        workgroupBarrier();
    }
    let mean_sq = scratch[0] / f32(params.ne0);
    let scale   = 1.0 / sqrt(mean_sq + params.eps);

    // Phase 3: dst = scale * rn_src * w + residual.
    // Note: in INPLACE_RN_DST mode this OVERWRITES rn_src_dst — but each
    // workgroup processes a different row, so no cross-workgroup race.
    // Within a row, all reads (in phase 1) complete before any write (phase 3),
    // bounded by workgroupBarrier at end of phase 2.
    col = lid.x;
    for (var k: u32 = 0u; k < elems_per_thread; k = k + 1u) {
        if (col >= params.ne0) {
            break;
        }
        let v = read_rn(row_rn + col);
        let w = mul_w[row_w + (col % params.w_ne0)];
        let r = residual[row_res + col];
        write_dst(row_d + col, scale * v * w + r);
        col = col + WG_SIZE;
    }
}
