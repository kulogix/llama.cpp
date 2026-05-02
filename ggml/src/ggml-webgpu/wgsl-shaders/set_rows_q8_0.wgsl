// [wllama-fork 2026-05-01] SET_ROWS shader for block_q8_0 destination.
//
// Quantizes f32 source rows into ggml's block_q8_0 format and writes them
// into the indexed dst rows. Unlocks q8_0 KV cache on the WebGPU backend
// for any model up to head_dim=512 (gemma-4 E2B), with explicit unroll
// for EPT={1,2}.
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
//   - Two compile-time variants:
//
//     EPT=1 (head_dim ≤ 256): WG_SIZE = head_dim. Each thread handles one
//                             element. Original simple path (bit-identical
//                             to pre-2026-05-01 shipping behavior).
//
//     EPT=2 (head_dim = 512): WG_SIZE = 256. Each thread handles 2 elements
//                             (strided: lane and lane+256). Two unrolled
//                             passes for phase-1 load, phase-2 reduction,
//                             phase-3 quantize. Lifts gemma-4's head_dim=512
//                             above the previous 256-element cap.
//
//   - Why explicit unrolling rather than a `for (e in 0..EPT)` loop: WGSL's
//     uniform-control-flow rule (required for workgroupBarrier) is checked
//     statically by Tint, which is conservative about loops containing
//     barriers. Even with a compile-constant bound (`for e < 2u`), Tint
//     rejected the construct with `'workgroupBarrier' must only be called
//     from uniform control flow`. Inlining the iterations explicitly puts
//     every barrier at the top level of the function, where uniformity is
//     trivially satisfied.
//
//   - EPT=3 (head_dim = 768) and EPT=4 (head_dim = 1024) would extend the
//     same pattern with two more unrolled blocks per phase, but no current
//     KV head_dim hits those — supports_op gates them out today. Add them
//     when a model needs them.
//
// row size analysis (head_dim is multiple of 32 for q8_0):
//   head_dim=64  -> 2 blocks, 68 bytes  (aligned, EPT=1)
//   head_dim=128 -> 4 blocks, 136 bytes (aligned, EPT=1)
//   head_dim=256 -> 8 blocks, 272 bytes (aligned, EPT=1, original path)
//   head_dim=512 -> 16 blocks, 544 bytes (aligned, EPT=2 — gemma-4 E2B)
//
// shared memory budget per workgroup (head_dim=512 worst case):
//   sh_abs:        f32 × 512 = 2048 bytes
//   sh_quant:      i32 × 512 = 2048 bytes
//   sh_scale_bits: u32 × 16  =   64 bytes
//   total:                      4160 bytes
//   WebGPU minimum maxComputeWorkgroupStorageSize is 16 KB (Safari/Firefox).
//   4× margin.

#ifdef HAS_F16
enable f16;
#endif

// HEAD_DIM cap for shared-memory arrays. Keep at 512 until a model needs
// EPT=3 or EPT=4 (then bump to 1024 + add the unrolled passes).
const MAX_HEAD_DIM: u32 = 512u;
const MAX_BLOCKS_PER_ROW: u32 = MAX_HEAD_DIM / 32u;

// ELEMS_PER_THREAD must be set per pipeline at compile time (1 or 2). If
// the build forgot to set it, default to 1 (back-compat with pre-rewrite
// behavior).
#ifndef ELEMS_PER_THREAD
#define ELEMS_PER_THREAD 1
#endif

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

// Inline helper: per-block 32-element pairwise absmax reduction. Reads/writes
// `sh_abs` in the [block_base, block_base+32) range. Each call places 5
// barriers (16→8→4→2→1) — these are the only barriers in phase 2, and they
// sit in the function body (not nested inside an outer loop) so Tint's
// uniformity analysis trivially accepts them.
//
// Note: this is an inline manually-expanded pattern, NOT a function call —
// WGSL doesn't support functions with workgroupBarrier in user code.
// We invoke this PATTERN once per pass via copy-paste below.

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
    // [wllama-fork 2026-05-01] On i64 overflow we set the host-readable error
    // flag and continue with garbage dst contents. The host checks `error`
    // after the dispatch completes and discards bad rows. We do NOT
    // early-return here: subsequent workgroupBarriers require uniform CF, and
    // Tint's analysis can't infer that the buffer-read driving the return is
    // workgroup-uniform. Continuing the work is wasted compute on the (rare)
    // overflow path but keeps the static uniformity proof simple.
    //
    // The atomicMax is deliberately unconditional so Tint cannot DCE the
    // `error` binding (we observed v3 stripping binding 3 from the layout
    // when the store was inside an `if (idx_hi != 0u)` block).
    atomicMax(&error, u32(idx[idx_high_off + 1u] != 0u));
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

    // ===== PHASE 1: load src + sh_abs (unrolled per ELEMS_PER_THREAD) =====
    //
    // STRIDED ownership: pass `e` of thread `lane` owns element
    // `e * WG_SIZE + lane`. For EPT=1 there's one pass; for EPT=2 there are
    // two unrolled passes covering the full 512 elements with WG_SIZE=256.
    //
    // The per-thread loaded values stay in scalar locals (x0, x1) so phase 3
    // doesn't need to re-read from src.
    let elem0 = lane;                        // pass 0: lane in [0, WG_SIZE)
    let v0 = src[src_row_off + elem0];
    sh_abs[elem0] = abs(v0);
#if ELEMS_PER_THREAD == 2
    let elem1 = WG_SIZE + lane;              // pass 1: lane in [WG_SIZE, 2*WG_SIZE)
    let v1 = src[src_row_off + elem1];
    sh_abs[elem1] = abs(v1);
#endif
    workgroupBarrier();

    // ===== PHASE 2: per-block 32-element absmax reduction (unrolled) =====
    //
    // Each block is 32 elements. WG_SIZE=256 covers blocks_per_pass = 8
    // blocks at a time. Within each pass, threads 0..31 reduce block 0,
    // threads 32..63 reduce block 1, etc.
    //
    // For EPT=1 (head_dim≤256) we have head_dim/32 blocks, with WG_SIZE = head_dim
    // → blocks_per_pass = head_dim/32 → one pass covers ALL blocks. (Original
    // behavior; bit-identical to pre-rewrite.)
    //
    // For EPT=2 (head_dim=512, WG_SIZE=256) we have 16 blocks total: pass 0
    // covers blocks 0..7, pass 1 covers blocks 8..15.
    //
    // The 5-stage pairwise reduction (stride 16→8→4→2→1) is itself a for-loop
    // with constant bounds — Tint accepts barriers in that loop because the
    // bound (16) is a literal AND the loop is at the top level of the function
    // (not nested inside another loop with barriers).
    let elem_in_block = lane % 32u;

    // ---- Pass 0: blocks 0 .. (WG_SIZE/32 - 1) ----
    {
        let block_in_row = lane / 32u;
        let block_base   = block_in_row * 32u;
        for (var stride = 16u; stride > 0u; stride >>= 1u) {
            if (elem_in_block < stride) {
                let other = sh_abs[block_base + elem_in_block + stride];
                sh_abs[block_base + elem_in_block] = max(sh_abs[block_base + elem_in_block], other);
            }
            workgroupBarrier();
        }
    }

#if ELEMS_PER_THREAD == 2
    // ---- Pass 1: blocks (WG_SIZE/32) .. (2*WG_SIZE/32 - 1) ----
    {
        let block_in_row = (WG_SIZE / 32u) + lane / 32u;
        let block_base   = block_in_row * 32u;
        for (var stride = 16u; stride > 0u; stride >>= 1u) {
            if (elem_in_block < stride) {
                let other = sh_abs[block_base + elem_in_block + stride];
                sh_abs[block_base + elem_in_block] = max(sh_abs[block_base + elem_in_block], other);
            }
            workgroupBarrier();
        }
    }
#endif

    // ===== PHASE 3: quantize + write per-block scale (unrolled) =====
    //
    // Each thread reads its block's amax from sh_abs[block_base], computes
    // scale + inv_scale, quantizes its element. Thread `elem_in_block == 0`
    // of each block writes the f16 scale's u16 bit pattern into
    // sh_scale_bits[block_idx].

    // ---- Pass 0 ----
    {
        let elem        = elem0;
        let block_idx   = elem / 32u;
        let elem_in_blk = elem % 32u;
        let block_base  = block_idx * 32u;
        let block_amax  = sh_abs[block_base];
        let scale_f32   = block_amax / 127.0;
#ifdef HAS_F16
        let scale_storage     = f16(scale_f32);
        let scale_for_dequant = f32(scale_storage);
#else
        let scale_storage     = scale_f32;
        let scale_for_dequant = scale_f32;
#endif
        let inv_scale = select(0.0, 1.0 / scale_for_dequant, block_amax > 0.0);
        let q = i32(round(v0 * inv_scale));
        sh_quant[elem] = clamp(q, -127, 127);

        if (elem_in_blk == 0u) {
#ifdef HAS_F16
            let packed = pack2x16float(vec2<f32>(scale_f32, 0.0));
            sh_scale_bits[block_idx] = packed & 0xffffu;
#else
            sh_scale_bits[block_idx] = bitcast<u32>(scale_f32) & 0xffffu;
#endif
        }
    }

#if ELEMS_PER_THREAD == 2
    // ---- Pass 1 ----
    {
        let elem        = elem1;
        let block_idx   = elem / 32u;
        let elem_in_blk = elem % 32u;
        let block_base  = block_idx * 32u;
        let block_amax  = sh_abs[block_base];
        let scale_f32   = block_amax / 127.0;
#ifdef HAS_F16
        let scale_storage     = f16(scale_f32);
        let scale_for_dequant = f32(scale_storage);
#else
        let scale_storage     = scale_f32;
        let scale_for_dequant = scale_f32;
#endif
        let inv_scale = select(0.0, 1.0 / scale_for_dequant, block_amax > 0.0);
        let q = i32(round(v1 * inv_scale));
        sh_quant[elem] = clamp(q, -127, 127);

        if (elem_in_blk == 0u) {
#ifdef HAS_F16
            let packed = pack2x16float(vec2<f32>(scale_f32, 0.0));
            sh_scale_bits[block_idx] = packed & 0xffffu;
#else
            sh_scale_bits[block_idx] = bitcast<u32>(scale_f32) & 0xffffu;
#endif
        }
    }
#endif

    workgroupBarrier();

    // ===== PHASE 4: lane 0 serializes the row's u32 writes =====
    //
    // Composes the 34-bytes-per-block layout (f16 scale + 32 i8s) into
    // u32 words and writes them into dst. Pure intra-workgroup writes,
    // no atomics needed (single thread). For head_dim=512 that's 136
    // u32 words per row.
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
