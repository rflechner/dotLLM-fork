#include <metal_stdlib>
using namespace metal;

// ── RMS Normalization ────────────────────────────────────────────────────────
//
// For each token (row), computes:
//   rms  = sqrt( mean(x²) + eps )
//   y[i] = x[i] / rms * weight[i]
//
// This kernel is fundamentally different from add/silu/RoPE:
//   → a whole threadgroup collaborates to process ONE row.
//   → threads must synchronize to share an intermediate result.
//
// CUDA → Metal mapping:
//   blockIdx.x            → threadgroup_position_in_grid   (row/token index)
//   threadIdx.x           → thread_position_in_threadgroup (lane within group)
//   blockDim.x            → threads_per_threadgroup        (group size)
//   __shared__ float ws[] → threadgroup float ws[]         (shared memory)
//   __syncthreads()       → threadgroup_barrier(...)       (sync barrier)
//   __shfl_down_sync()    → simd_sum()                    (warp reduction)
//   rsqrtf()              → rsqrt()

kernel void rmsnorm_f32(
    device const float* input   [[ buffer(0) ]],   // x  : [seqLen, n]
    device const float* weight  [[ buffer(1) ]],   // w  : [n]  (RMSNorm scale)
    device       float* output  [[ buffer(2) ]],   // y  : [seqLen, n]
    constant     int&   n       [[ buffer(3) ]],   // hidden dimension (e.g. 4096)
    constant     float& eps     [[ buffer(4) ]],   // numerical stability epsilon (e.g. 1e-5)
    uint row    [[ threadgroup_position_in_grid ]],   // which token? (= blockIdx.x)
    uint lid    [[ thread_position_in_threadgroup ]], // lane within group (= threadIdx.x)
    uint tgSize [[ threads_per_threadgroup ]])         // threads per group (= blockDim.x)
{
    // Memory shared across all threads in the group.
    // 32 slots cover up to tgSize=1024: 1024/32 = 32 simd groups max.
    threadgroup float simd_sums[32]; // partial sums per simd group
    threadgroup float ri = 0.0f;    // final result: 1/rms, broadcast to all threads
                                    // (initialized to suppress the uninitialized warning;
                                    //  the barrier guarantees thread 0 writes it before phase 6)

    // Pointers to the start of this row (this token).
    // Each threadgroup processes a different row → offset = row * n.
    const device float* x = input  + (size_t)row * n;
    device       float* y = output + (size_t)row * n;

    // ── Phase 1: partial sum of squares ──────────────────────────────────────
    // Each thread accumulates the elements assigned to it:
    //   lid, lid+tgSize, lid+2*tgSize, ...
    // Example: n=4096, tgSize=256, lid=3 → processes x[3], x[259], x[515], ...
    float sum_sq = 0.0f;
    for (uint i = lid; i < (uint)n; i += tgSize) {
        float v = x[i];
        sum_sq += v * v;
    }

    // ── Phase 2: reduction within the simd group (= CUDA warp) ───────────────
    // simd_sum() adds sum_sq across all 32 threads in the simd group.
    // CUDA equivalent: the __shfl_down_sync(... off >>= 1) loop.
    // After this call, every thread in the simd group sees the group's total.
    sum_sq = simd_sum(sum_sq);

    // ── Phase 3: cross-simd reduction via threadgroup memory ─────────────────
    // Only lane 0 of each simd group stores its sum into simd_sums[].
    // simd_lane = position within the simd group (0..31)
    // simd_id   = index of the simd group within the threadgroup
    uint simd_lane = lid % 32;
    uint simd_id   = lid / 32;
    if (simd_lane == 0) {
        simd_sums[simd_id] = sum_sq;
    }

    // All threads wait until simd_sums[] is fully written.
    // CUDA equivalent: __syncthreads()
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 4: first 32 threads reduce simd_sums[] ─────────────────────────
    // Re-use simd_sum() across the (up to 32) entries in simd_sums[].
    // One simd group is enough for this (32 threads, 32 values max).
    if (lid < 32) {
        uint num_simds = (tgSize + 31) / 32;               // number of active simd groups
        sum_sq = (lid < num_simds) ? simd_sums[lid] : 0.0f;
        sum_sq = simd_sum(sum_sq);                         // → thread 0 holds the grand total
    }

    // ── Phase 5: thread 0 computes ri and stores it in threadgroup memory ────
    // rsqrt(x) = 1 / sqrt(x)   (CUDA equivalent: rsqrtf())
    if (lid == 0) {
        ri = rsqrt(sum_sq / (float)n + eps);
    }

    // All threads wait until `ri` is written before reading it.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 6: normalize and scale ─────────────────────────────────────────
    // All threads now share `ri`.
    // Each thread processes the same elements as in Phase 1.
    float scale = ri;
    for (uint i = lid; i < (uint)n; i += tgSize) {
        y[i] = x[i] * scale * weight[i];
    }
}
