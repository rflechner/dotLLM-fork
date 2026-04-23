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
// Port of quantized_gemv.cu::quantized_gemv_q8_0
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
        device const char* qs = (device const char*)(block + 2);

        float s = 0.0f;
        for (int j = 0; j < 32; j++)
            s += float(qs[j]) * float(x[b * 32 + j]);
        acc += d * s;
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

    for (int sb = (int)tid; sb < sbpr; sb += (int)tgw)
    {
        device const uchar* block      = w_row + sb * 144;
        float               d          = float(*(device const half*)(block));
        float               dmin       = float(*(device const half*)(block + 2));
        device const uchar* scales_raw = block + 4;
        device const uchar* qs         = block + 16;

        // 4 pairs of sub-blocks, each pair shares 32 qs bytes
        for (int pair = 0; pair < 4; pair++)
        {
            int sb_even = pair * 2;
            int sb_odd  = pair * 2 + 1;

            int sc0, m0, sc1, m1;
            if (sb_even < 4)
            {
                sc0 = (int)(scales_raw[sb_even]     & 0x3Fu);
                m0  = (int)(scales_raw[sb_even + 4] & 0x3Fu);
                sc1 = (int)(scales_raw[sb_odd]      & 0x3Fu);
                m1  = (int)(scales_raw[sb_odd  + 4] & 0x3Fu);
            }
            else
            {
                sc0 = (int)((scales_raw[sb_even + 4] & 0x0Fu) | ((scales_raw[sb_even - 4] >> 6) << 4));
                m0  = (int)((scales_raw[sb_even + 4] >> 4)    | ((scales_raw[sb_even]     >> 6) << 4));
                sc1 = (int)((scales_raw[sb_odd  + 4] & 0x0Fu) | ((scales_raw[sb_odd  - 4] >> 6) << 4));
                m1  = (int)((scales_raw[sb_odd  + 4] >> 4)    | ((scales_raw[sb_odd]      >> 6) << 4));
            }

            float scale0 = d * float(sc0);
            float min0   = dmin * float(m0);
            float scale1 = d * float(sc1);
            float min1   = dmin * float(m1);

            device const uchar* pair_qs   = qs + pair * 32;
            int                 base_x    = sb * 256 + pair * 64;

            for (int j = 0; j < 32; j++)
            {
                uint  bv     = pair_qs[j];
                float x_even = float(x[base_x + j]);
                float x_odd  = float(x[base_x + j + 32]);
                acc += (scale0 * float(bv & 0x0Fu) - min0) * x_even;
                acc += (scale1 * float(bv >> 4)    - min1) * x_odd;
            }
        }
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
// Port of quantized_gemv.cu::quantized_gemv_q6_k
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

            for (int l = 0; l < 32; l++)
            {
                int isc = l / 16;

                int q1 = (int)((ql_half[l]      & 0x0Fu) | (((uint)(qh_half[l] >> 0) & 3u) << 4)) - 32;
                int q2 = (int)((ql_half[l + 32] & 0x0Fu) | (((uint)(qh_half[l] >> 2) & 3u) << 4)) - 32;
                int q3 = (int)((ql_half[l]       >> 4)   | (((uint)(qh_half[l] >> 4) & 3u) << 4)) - 32;
                int q4 = (int)((ql_half[l + 32]  >> 4)   | (((uint)(qh_half[l] >> 6) & 3u) << 4)) - 32;

                float s0 = d * float(sc_half[isc]);
                float s1 = d * float(sc_half[isc + 2]);
                float s2 = d * float(sc_half[isc + 4]);
                float s3 = d * float(sc_half[isc + 6]);

                acc += s0 * float(q1) * float(x[x_off + l]);
                acc += s1 * float(q2) * float(x[x_off + l + 32]);
                acc += s2 * float(q3) * float(x[x_off + l + 64]);
                acc += s3 * float(q4) * float(x[x_off + l + 96]);
            }
        }
    }

    threadgroup float ws[32];
    WARP_REDUCE(acc, tid, tgw, ws, row, y)
}
