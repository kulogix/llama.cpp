// [wllama-fork 2026-05-01] SET_ROWS shader for block_q8_0 destination.
//
// Quantizes f32 source rows into ggml's block_q8_0 format and writes them
// into the indexed dst rows. Unlocks q8_0 KV cache on the WebGPU backend
// (previously SET_ROWS only handled f16/f32 dst, forcing wllama to use
// f16 KV — see src/wllama.ts:181 + cpp/wllama.cpp KV-quant policy).
//
// q-quant recipe (block_q8_0):
//   layout: { ggml_half d; int8_t qs[32]; } = 34 bytes
//   recipe:
//     1. amax = max(|x[0..31]|)
//     2. d_f32 = amax / 127                       (scale in f32)
//     3. d_f16 = f16(d_f32)                       (truncate to f16 for storage)
//     4. inv_d = 1.0 / f32(d_f16)                 (use the f16-truncated value
//                                                  for the per-element divisor
//                                                  so dequant uses the SAME scale
//                                                  the writer used; matches the
//                                                  CUDA / Metal reference impls,
//                                                  see ggml-metal.metal:356-375
//                                                  and cpy-utils.cuh:137-154)
//     5. q[i] = clamp(round(x[i] * inv_d), -127, 127)   (note: -127, NOT -128;
//                                                        symmetric range, matches
//                                                        ggml's CPU quantize)
//     6. write d_f16 + 32 i8s into the 34-byte block
//
// Dispatch model:
//   - 1 workgroup per dst row.
//   - workgroup_size = HEAD_DIM (one thread per element). For typical KV
//     head_dim = 64..256 this fits comfortably under WGSL's 256-thread max.
//   - All threads quantize in parallel; thread 0 serializes the 17-68
//     u32-word writes for the row (avoids the 34-byte misalignment race
//     since blocks span u32 boundaries).
//
// row size analysis (head_dim is multiple of 32 for q8_0):
//   head_dim=32  -> 1 block,  34 bytes  (ROW NOT u32-ALIGNED — unsupported)
//   head_dim=64  -> 2 blocks, 68 bytes  (aligned, ok)
//   head_dim=128 -> 4 blocks, 136 bytes (aligned, ok)
//   head_dim=256 -> 8 blocks, 272 bytes (aligned, ok)
//   head_dim>=64 always aligns since (head_dim/32) * 34 is multiple of 4
//   when head_dim/32 is even. We require head_dim % 64 == 0 for safety.

#ifdef HAS_F16
enable f16;
#endif

// HEAD_DIM is set per pipeline at compile time (= context.dst->ne[0]).
// MAX_BLOCKS_PER_ROW caps shared-memory size; we support up to head_dim=256.
const MAX_HEAD_DIM: u32 = 256u;
const MAX_BLOCKS_PER_ROW: u32 = MAX_HEAD_DIM / 32u;

@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> idx: array<u32>;

@group(0) @binding(2)
var<storage, read_write> dst: array<u32>;

#ifdef I64_IDX
@group(0) @binding(3)
var<storage, read_write> error: atomic<u32>;
#define PARAMS_BINDING 4
#else
#define PARAMS_BINDING 3
#endif

struct Params {
    offset_src:  u32, // in elements (f32)
    offset_idx:  u32, // in elements (u32 / u64)
    offset_dst:  u32, // in BYTES (q8_0 block layout is byte-addressed)

    // src strides (in elements)
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    // idx strides (in elements)
    stride_idx0: u32,
    stride_idx1: u32,
    stride_idx2: u32,

    // dst strides (in BYTES; row-stride must be multiple of 4)
    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,

    // src shape
    ne0:    u32, // = head_dim, MUST equal HEAD_DIM compile constant
    n_rows: u32,
    ne2:    u32,
    ne3:    u32,

    // idx shape
    idx1: u32,
    idx2: u32,
};

@group(0) @binding(PARAMS_BINDING)
var<uniform> params: Params;

// Workgroup-shared buffers for the quantize/reduce pipeline.
var<workgroup> sh_abs:   array<f32, MAX_HEAD_DIM>;
var<workgroup> sh_quant: array<i32, MAX_HEAD_DIM>;
var<workgroup> sh_scale_bits: array<u32, MAX_BLOCKS_PER_ROW>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id)        wg_id:    vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
    let lane = local_id.x;
    let head_dim = params.ne0;
    let row_global = wg_id.x;

    // row_global enumerates ne3 * ne2 * n_rows. Decompose:
    let rows_per_n2 = params.n_rows;
    let i_src3 = row_global / (params.ne2 * rows_per_n2);
    var rem = row_global % (params.ne2 * rows_per_n2);
    let i_src2 = rem / rows_per_n2;
    let i_src1 = rem % rows_per_n2;

    // Resolve idx (= dst row index within the layer).
    let i_idx2 = i_src3 % params.idx2;
    let i_idx1 = i_src2 % params.idx1;
    let i_idx0 = i_src1;

#ifdef I64_IDX
    let idx_high_off = (params.offset_idx
                        + i_idx0 * params.stride_idx0
                        + i_idx1 * params.stride_idx1
                        + i_idx2 * params.stride_idx2) * 2u;
    let idx_val = idx[idx_high_off];
    let idx_hi  = idx[idx_high_off + 1u];
    if (idx_hi != 0u) {
        atomicStore(&error, 1u);
        return;
    }
#else
    let idx_off = params.offset_idx
                  + i_idx0 * params.stride_idx0
                  + i_idx1 * params.stride_idx1
                  + i_idx2 * params.stride_idx2;
    let idx_val = idx[idx_off];
#endif

    // Source element offset for this thread.
    let src_row_off = params.offset_src
                      + i_src1 * params.stride_src1
                      + i_src2 * params.stride_src2
                      + i_src3 * params.stride_src3;

    // Skip threads outside head_dim (workgroup may be padded up).
    let active = lane < head_dim;

    // Load src element + record |x| for the block-amax reduction.
    var x: f32 = 0.0;
    if (active) {
        x = src[src_row_off + lane];
    }
    sh_abs[lane] = abs(x);
    workgroupBarrier();

    // Each thread participates in its own block's 32-element absmax reduction.
    // Iterative pairwise max within shared memory — works regardless of
    // subgroup size or whether subgroups are even available.
    let block_in_row = lane / 32u;
    let elem_in_block = lane % 32u;
    let block_base = block_in_row * 32u;

    // Stage reduction: 32 -> 16 -> 8 -> 4 -> 2 -> 1
    for (var stride = 16u; stride > 0u; stride >>= 1u) {
        if (elem_in_block < stride) {
            let other = sh_abs[block_base + elem_in_block + stride];
            sh_abs[block_base + elem_in_block] = max(sh_abs[block_base + elem_in_block], other);
        }
        workgroupBarrier();
    }
    let block_amax = sh_abs[block_base];

    // Compute scale per the recipe: f32 -> f16-truncate -> use f16 value.
    let scale_f32 = block_amax / 127.0;
#ifdef HAS_F16
    let scale_storage = f16(scale_f32);
    let scale_for_dequant = f32(scale_storage);
#else
    // No f16 in shader: truncate manually via bitcast round-trip is messy.
    // Best-effort: use the f32 scale unchanged (slight read/write mismatch).
    let scale_storage = scale_f32;
    let scale_for_dequant = scale_f32;
#endif
    let inv_scale = select(0.0, 1.0 / scale_for_dequant, block_amax > 0.0);

    // Quantize this thread's element to i8 (range [-127, 127]).
    if (active) {
        let q = i32(round(x * inv_scale));
        sh_quant[lane] = clamp(q, -127, 127);
    } else {
        sh_quant[lane] = 0;
    }

    // Thread 0 of each block writes the f16 scale's u16 bit pattern into
    // sh_scale_bits[block_in_row], so the row writer can splice it later.
    if (elem_in_block == 0u && active) {
#ifdef HAS_F16
        // pack2x16float packs (lo, hi) f32 -> 2x f16 LSB pattern in low / high
        // 16 bits respectively. We want the SAME f16 truncation we used for
        // dequant. Going via bitcast<u32>(vec2(x, 0.0)) gives us the canonical
        // IEEE 754 half encoding in low 16 bits.
        let packed = pack2x16float(vec2<f32>(scale_f32, 0.0));
        sh_scale_bits[block_in_row] = packed & 0xffffu;
#else
        // f32-fallback path: store low 16 bits of f32 (DOES NOT round-trip;
        // marker only).
        sh_scale_bits[block_in_row] = bitcast<u32>(scale_f32) & 0xffffu;
#endif
    }
    workgroupBarrier();

    // Thread 0 serializes the row's u32 writes (avoids the 34-byte block-
    // misalignment race; pure intra-workgroup writes, no atomics needed).
    if (lane == 0u) {
        let dst_row_byte_base = params.offset_dst
                                + idx_val * params.stride_dst1
                                + i_src2  * params.stride_dst2
                                + i_src3  * params.stride_dst3;
        let dst_row_word_base = dst_row_byte_base / 4u;

        let blocks_per_row = head_dim / 32u;
        let row_byte_count = blocks_per_row * 34u;

        // Walk u32 words; for each, compose its 4 bytes from per-block scale
        // and quantized values. For each row byte b in [0, row_byte_count):
        //   block_idx = b / 34
        //   in_block  = b % 34
        //   in_block == 0 -> scale low byte
        //   in_block == 1 -> scale high byte
        //   else          -> qs[block_idx*32 + (in_block - 2)]
        let n_words = (row_byte_count + 3u) / 4u;
        for (var w: u32 = 0u; w < n_words; w = w + 1u) {
            var word_val: u32 = 0u;
            for (var b_in_word: u32 = 0u; b_in_word < 4u; b_in_word = b_in_word + 1u) {
                let row_byte = w * 4u + b_in_word;
                if (row_byte >= row_byte_count) {
                    continue;
                }
                let block_idx = row_byte / 34u;
                let in_block  = row_byte % 34u;
                var byte_val: u32 = 0u;
                if (in_block == 0u) {
                    byte_val = sh_scale_bits[block_idx] & 0xffu;
                } else if (in_block == 1u) {
                    byte_val = (sh_scale_bits[block_idx] >> 8u) & 0xffu;
                } else {
                    byte_val = u32(sh_quant[block_idx * 32u + (in_block - 2u)]) & 0xffu;
                }
                word_val = word_val | (byte_val << (b_in_word * 8u));
            }
            dst[dst_row_word_base + w] = word_val;
        }
    }
}
