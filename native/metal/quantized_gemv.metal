// Quantized GEMV kernels for dotLLM decode path — FP16 input/output.
// y[n] = W[n,k] @ x[k]  where W is quantized, x is FP16, y is FP16.
// FP32 accumulation throughout; y written as half on completion.
// One threadgroup per output row; threads stride over blocks, then reduce.
// Port of quantized_gemv.cu

#include <metal_stdlib>
using namespace metal;

// ── Shared reduction macro ────────────────────────────────────────────────────
// Identical pattern to quantized_gemv_q8_0_f32in:
//   1. simd_sum within each simd-group (broadcasts to all lanes).
//   2. Lane 0 of each simd-group writes to threadgroup scratch ws[].
//   3. Simd-group 0 reduces ws[] and stores result in y[row].
#define WARP_REDUCE(acc, tid, tgw, ws, row, y)          \
{                                                         \
    uint _lane = (tid) % 32u;                             \
    uint _wid  = (tid) / 32u;                             \
    (acc) = simd_sum(acc);                                \
    if (_lane == 0) (ws)[_wid] = (acc);                   \
    threadgroup_barrier(mem_flags::mem_threadgroup);      \
    if (_wid == 0) {                                      \
        uint _nw = ((tgw) + 31u) / 32u;                  \
        (acc) = (_lane < _nw) ? (ws)[_lane] : 0.0f;      \
        (acc) = simd_sum(acc);                            \
    }                                                     \
    if ((tid) == 0) (y)[(row)] = half(acc);               \
}

// ── quantized_gemv_q8_0 ──────────────────────────────────────────────────────
// Q8_0: 34 bytes per 32 values (2-byte half scale + 32 int8 weights).
//
// Vectorized inner loop: 8 × float4 FMAs instead of 32 scalar FMAs.
//   - x[b*32..b*32+31] is 64-byte aligned (b*32*sizeof(half) = b*64) → safe half4 loads.
//   - weight quants start at block+2 (2-byte aligned, not always 4-byte). Apple
//     GPUs accept the misalignment at a minor cost; this is still net positive
//     because we issue 4× fewer load instructions and 4× fewer FMAs.
kernel void quantized_gemv_q8_0(
    device const uchar* weight [[buffer(0)]],
    device const half*  x      [[buffer(1)]],
    device       half*  y      [[buffer(2)]],
    constant int&       n      [[buffer(3)]],
    constant int&       k      [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgw [[threads_per_threadgroup]])
{
    if ((int)row >= n) return;

    const int bpr = k / 32;
    device const uchar* w_row = weight + (ulong)row * bpr * 34;
    float acc = 0.0f;

    for (int b = (int)tid; b < bpr; b += (int)tgw)
    {
        device const uchar* block = w_row + b * 34;
        float d = float(*(device const half*)block);

        // 32 int8 quants as 8 × packed_char4 (alignment 1 — `char4` requires
        // 4-byte alignment, which `block + 2` does NOT satisfy for arbitrary rows).
        device const packed_char4* qs4 = (device const packed_char4*)(block + 2);
        // 32 half activations as 8 × half4. x base is 16-byte aligned and
        // b*32*sizeof(half) = b*64 → always 8-byte aligned. Safe.
        device const half4* x4  = (device const half4*)(x + b * 32);

        float4 s4 = float4(0.0f);
        for (int j = 0; j < 8; j++) {
            float4 qf = float4(char4(qs4[j]));   // packed → char4 → signed widen
            float4 xf = float4(x4[j]);
            s4 = fma(qf, xf, s4);
        }
        acc += d * (s4.x + s4.y + s4.z + s4.w);
    }

    threadgroup float ws[32];
    WARP_REDUCE(acc, tid, tgw, ws, row, y)
}

// ── quantized_gemv_q5_0 ──────────────────────────────────────────────────────
// Q5_0: 22 bytes per 32 values (2-byte half d + 4-byte qh + 16 packed nibbles).
// Port of quantized_gemv.cu::quantized_gemv_q5_0
kernel void quantized_gemv_q5_0(
    device const uchar* weight [[buffer(0)]],
    device const half*  x      [[buffer(1)]],
    device       half*  y      [[buffer(2)]],
    constant int&       n      [[buffer(3)]],
    constant int&       k      [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgw [[threads_per_threadgroup]])
{
    if ((int)row >= n) return;

    const int bpr = k / 32;
    device const uchar* w_row = weight + (ulong)row * bpr * 22;
    float acc = 0.0f;

    for (int b = (int)tid; b < bpr; b += (int)tgw)
    {
        device const uchar* block = w_row + b * 22;
        float d = float(*(device const half*)block);
        // Read qh as 4 bytes (Q5_0 blocks are 22 bytes — may be unaligned)
        uint qh = (uint)block[2] | ((uint)block[3] << 8) |
                  ((uint)block[4] << 16) | ((uint)block[5] << 24);
        device const uchar* qs = block + 6;

        float s = 0.0f;
        for (int j = 0; j < 16; j++)
        {
            uint packed = qs[j];
            int lo = (int)((packed & 0x0Fu) | (((qh >> j)      & 1u) << 4));
            int hi = (int)((packed >> 4)    | (((qh >> (j+16)) & 1u) << 4));
            s += float(lo - 16) * float(x[b * 32 + j]);
            s += float(hi - 16) * float(x[b * 32 + j + 16]);
        }
        acc += d * s;
    }

    threadgroup float ws[32];
    WARP_REDUCE(acc, tid, tgw, ws, row, y)
}

// ── quantized_gemv_q4_k ──────────────────────────────────────────────────────
// Q4_K: 144 bytes per 256 values (superblock: d, dmin, 12 scales, 128 nibbles).
// Port of quantized_gemv.cu::quantized_gemv_q4_k
kernel void quantized_gemv_q4_k(
    device const uchar* weight [[buffer(0)]],
    device const half*  x      [[buffer(1)]],
    device       half*  y      [[buffer(2)]],
    constant int&       n      [[buffer(3)]],
    constant int&       k      [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgw [[threads_per_threadgroup]])
{
    if ((int)row >= n) return;

    const int sbpr = k / 256;
    device const uchar* w_row = weight + (ulong)row * sbpr * 144;
    float acc = 0.0f;

    // Parallelize across individual coefficients instead of assigning only one
    // thread per 256-value superblock. For common decode shapes such as k=4096,
    // sbpr is only 16, so the old loop used only 16 useful threads out of a
    // 256-thread group. This keeps the same one-threadgroup-per-output-row model,
    // but all lanes now contribute to the dot product.
    for (int idx = (int)tid; idx < k; idx += (int)tgw)
    {
        int sb       = idx >> 8;       // idx / 256
        int in_sb    = idx & 255;      // idx % 256
        int sub      = in_sb >> 5;     // 8 sub-blocks of 32 values
        int i        = in_sb & 31;     // element inside the 32-value sub-block
        int pair     = sub >> 1;       // two sub-blocks share one 32-byte qs stream

        device const uchar* block      = w_row + sb * 144;
        float               d          = float(*(device const half*)(block));
        float               dmin       = float(*(device const half*)(block + 2));
        device const uchar* scales_raw = block + 4;
        device const uchar* qs         = block + 16;

        int sc, m;
        if (sub < 4)
        {
            sc = (int)(scales_raw[sub]     & 0x3Fu);
            m  = (int)(scales_raw[sub + 4] & 0x3Fu);
        }
        else
        {
            sc = (int)((scales_raw[sub + 4] & 0x0Fu) | ((scales_raw[sub - 4] >> 6) << 4));
            m  = (int)((scales_raw[sub + 4] >> 4)    | ((scales_raw[sub]     >> 6) << 4));
        }

        uchar packed = qs[pair * 32 + i];
        int q = ((sub & 1) == 0) ? (int)(packed & 0x0Fu)
                                 : (int)(packed >> 4);

        acc += (d * float(sc) * float(q) - dmin * float(m)) * float(x[idx]);
    }

    threadgroup float ws[32];
    WARP_REDUCE(acc, tid, tgw, ws, row, y)
}

// ── quantized_gemv_q5_k ──────────────────────────────────────────────────────
// Q5_K: 176 bytes per 256 values (d, dmin, 12 scales, 32 qh, 128 qs).
// Port of quantized_gemv.cu::quantized_gemv_q5_k
kernel void quantized_gemv_q5_k(
    device const uchar* weight [[buffer(0)]],
    device const half*  x      [[buffer(1)]],
    device       half*  y      [[buffer(2)]],
    constant int&       n      [[buffer(3)]],
    constant int&       k      [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgw [[threads_per_threadgroup]])
{
    if ((int)row >= n) return;

    const int sbpr = k / 256;
    device const uchar* w_row = weight + (ulong)row * sbpr * 176;
    float acc = 0.0f;

    for (int sb = (int)tid; sb < sbpr; sb += (int)tgw)
    {
        device const uchar* block      = w_row + sb * 176;
        float               d          = float(*(device const half*)(block));
        float               dmin       = float(*(device const half*)(block + 2));
        device const uchar* scales_raw = block + 4;
        device const uchar* qh         = block + 16;
        device const uchar* qs         = block + 48;
        int                 base_x     = sb * 256;

        // 8 sub-blocks of 32 elements each.
        // qs layout (llama.cpp): sub-blocks 0&1 share qs[0..31], 2&3 share qs[32..63], etc.
        //   Even sub (nibbleHalf=0) → lo nibble; odd sub (nibbleHalf=1) → hi nibble.
        // qh layout: qh[i] has one bit per sub-block; bit5 for (sub, i) = (qh[i] >> sub) & 1.
        for (int sub = 0; sub < 8; sub++)
        {
            int sc, m;
            if (sub < 4)
            {
                sc = (int)(scales_raw[sub]     & 0x3Fu);
                m  = (int)(scales_raw[sub + 4] & 0x3Fu);
            }
            else
            {
                sc = (int)((scales_raw[sub + 4] & 0x0Fu) | ((scales_raw[sub - 4] >> 6) << 4));
                m  = (int)((scales_raw[sub + 4] >> 4)    | ((scales_raw[sub]     >> 6) << 4));
            }

            float scale   = d * float(sc);
            float min_val = dmin * float(m);

            int nibbleHalf = sub & 1;
            device const uchar* pair_qs = qs + (sub / 2) * 32;
            int x_off = base_x + sub * 32;

            for (int i = 0; i < 32; i++)
            {
                int lo4  = nibbleHalf == 0 ? (int)(pair_qs[i] & 0x0Fu)
                                           : (int)(pair_qs[i] >> 4);
                int bit5 = (int)((qh[i] >> sub) & 1u);
                acc += (scale * float(lo4 | (bit5 << 4)) - min_val) * float(x[x_off + i]);
            }
        }
    }

    threadgroup float ws[32];
    WARP_REDUCE(acc, tid, tgw, ws, row, y)
}

// ── quantized_gemv_q6_k ──────────────────────────────────────────────────────
// Q6_K: 210 bytes per 256 values (128 ql + 64 qh + 16 scales(int8) + 2 d).
//
// Vectorized inner loop: each iteration processes 4 consecutive `l` values
// using char4 / half4 ops. The four scale streams (s0..s3) and the four
// x-windows ([l..l+3], [l+32..l+35], [l+64..l+67], [l+96..l+99]) all read
// 4 contiguous halves at once → 4× fewer load instructions.
//
// `isc` (= l/16) splits each half-superblock at l=16. Within each l-batch of
// 4, all four iterations share the same isc (since 4 < 16), so we only need
// to refetch scales at l=0 and l=16.
kernel void quantized_gemv_q6_k(
    device const uchar* weight [[buffer(0)]],
    device const half*  x      [[buffer(1)]],
    device       half*  y      [[buffer(2)]],
    constant int&       n      [[buffer(3)]],
    constant int&       k      [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgw [[threads_per_threadgroup]])
{
    if ((int)row >= n) return;

    const int sbpr = k / 256;
    device const uchar* w_row = weight + (ulong)row * sbpr * 210;
    float acc = 0.0f;

    for (int sb = (int)tid; sb < sbpr; sb += (int)tgw)
    {
        device const uchar* block  = w_row + sb * 210;
        device const uchar* ql     = block;
        device const uchar* qh     = block + 128;
        device const char*  scales = (device const char*)(block + 192);
        float               d      = float(*(device const half*)(block + 208));
        int                 base_x = sb * 256;

        // Two 128-element halves, matching Q6_K GGUF layout
        for (int half_idx = 0; half_idx < 2; half_idx++)
        {
            device const uchar* ql_half = ql + half_idx * 64;
            device const uchar* qh_half = qh + half_idx * 32;
            device const char*  sc_half = scales + half_idx * 8;
            int                 x_off   = base_x + half_idx * 128;

            // Process l in batches of 4. isc = l/16 changes only at l=0 and l=16.
            for (int l = 0; l < 32; l += 4)
            {
                int isc = l / 16;
                float s0 = d * float(sc_half[isc]);
                float s1 = d * float(sc_half[isc + 2]);
                float s2 = d * float(sc_half[isc + 4]);
                float s3 = d * float(sc_half[isc + 6]);

                // 4 ql bytes for indices l..l+3, and another 4 at l+32..l+35.
                // packed_uchar4 has alignment 1 — required because ql_half + l can
                // land at an odd byte offset within a misaligned superblock base.
                uchar4 ql_lo4 = uchar4(*(device const packed_uchar4*)(ql_half + l));
                uchar4 ql_hi4 = uchar4(*(device const packed_uchar4*)(ql_half + l + 32));
                uchar4 qh_4   = uchar4(*(device const packed_uchar4*)(qh_half + l));

                // Decode 4 quants per stream: each is 4 bits from ql plus 2 bits from qh.
                int4 q1_v = int4(
                    (int)((ql_lo4.x & 0x0Fu) | (((uint)(qh_4.x >> 0) & 3u) << 4)),
                    (int)((ql_lo4.y & 0x0Fu) | (((uint)(qh_4.y >> 0) & 3u) << 4)),
                    (int)((ql_lo4.z & 0x0Fu) | (((uint)(qh_4.z >> 0) & 3u) << 4)),
                    (int)((ql_lo4.w & 0x0Fu) | (((uint)(qh_4.w >> 0) & 3u) << 4))) - 32;
                int4 q2_v = int4(
                    (int)((ql_hi4.x & 0x0Fu) | (((uint)(qh_4.x >> 2) & 3u) << 4)),
                    (int)((ql_hi4.y & 0x0Fu) | (((uint)(qh_4.y >> 2) & 3u) << 4)),
                    (int)((ql_hi4.z & 0x0Fu) | (((uint)(qh_4.z >> 2) & 3u) << 4)),
                    (int)((ql_hi4.w & 0x0Fu) | (((uint)(qh_4.w >> 2) & 3u) << 4))) - 32;
                int4 q3_v = int4(
                    (int)((ql_lo4.x >> 4)    | (((uint)(qh_4.x >> 4) & 3u) << 4)),
                    (int)((ql_lo4.y >> 4)    | (((uint)(qh_4.y >> 4) & 3u) << 4)),
                    (int)((ql_lo4.z >> 4)    | (((uint)(qh_4.z >> 4) & 3u) << 4)),
                    (int)((ql_lo4.w >> 4)    | (((uint)(qh_4.w >> 4) & 3u) << 4))) - 32;
                int4 q4_v = int4(
                    (int)((ql_hi4.x >> 4)    | (((uint)(qh_4.x >> 6) & 3u) << 4)),
                    (int)((ql_hi4.y >> 4)    | (((uint)(qh_4.y >> 6) & 3u) << 4)),
                    (int)((ql_hi4.z >> 4)    | (((uint)(qh_4.z >> 6) & 3u) << 4)),
                    (int)((ql_hi4.w >> 4)    | (((uint)(qh_4.w >> 6) & 3u) << 4))) - 32;

                // 4 contiguous half loads per stream (16 bytes each).
                float4 x0 = float4(*(device const half4*)(x + x_off + l));
                float4 x1 = float4(*(device const half4*)(x + x_off + l + 32));
                float4 x2 = float4(*(device const half4*)(x + x_off + l + 64));
                float4 x3 = float4(*(device const half4*)(x + x_off + l + 96));

                // Each stream contributes scale * dot(q, x).
                float4 q1f = float4(q1_v);
                float4 q2f = float4(q2_v);
                float4 q3f = float4(q3_v);
                float4 q4f = float4(q4_v);

                acc += s0 * dot(q1f, x0);
                acc += s1 * dot(q2f, x1);
                acc += s2 * dot(q3f, x2);
                acc += s3 * dot(q4f, x3);
            }
        }
    }

    threadgroup float ws[32];
    WARP_REDUCE(acc, tid, tgw, ws, row, y)
}
