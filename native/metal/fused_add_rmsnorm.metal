#include <metal_stdlib>
using namespace metal;

kernel void fused_add_rmsnorm_f16(
    device half* residual                   [[ buffer(0) ]],
    device const half* x                    [[ buffer(1) ]],
    device const half* weight               [[ buffer(2) ]],
    device half* output                     [[ buffer(3) ]],
    constant int& n                         [[ buffer(4) ]],
    constant float& eps                     [[ buffer(5) ]],
    
    uint block_id_x                         [[ threadgroup_position_in_grid ]],   // blockIdx.x
    uint threadIdx_x                        [[ thread_position_in_threadgroup ]], // threadIdx.x
    uint blockDim_x                         [[ threads_per_threadgroup ]],        // blockDim.x
    uint simd_group_width                   [[ threads_per_simdgroup]])
{
    const int row = block_id_x;
    device half* res_row = residual + (size_t)row * n;
    device const half* x_row = x + (size_t)row * n;
    device half* out_row = output + (size_t)row * n;
    
    // Pass 1: compute sum in FP32, write sum back to residual (FP16),
    // and accumulate sum-of-squares for RMS
    float sum_sq = 0.0f;
    for (int i = threadIdx_x; i < n; i += blockDim_x)
    {
        float r = float(res_row[i]);
        float xi = float(x_row[i]);
        float sum = r + xi;

        // Update residual with the sum (FP16 storage)
        res_row[i] = half(sum);

        // Accumulate for RMS from the FP32 sum (no extra truncation!)
        sum_sq += sum * sum;
    }

    // Warp reduction for sum_sq
    for (int offset = simd_group_width / 2; offset > 0; offset >>= 1)
        sum_sq += simd_shuffle_down(sum_sq, offset);

    threadgroup float warp_sums[32];
    int lane = threadIdx_x % simd_group_width;
    int warp_id = threadIdx_x / simd_group_width;

    if (lane == 0) warp_sums[warp_id] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (warp_id == 0)
    {
        int num_warps = (blockDim_x + simd_group_width - 1) / simd_group_width;
        sum_sq = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = simd_group_width / 2; offset > 0; offset >>= 1)
            sum_sq += simd_shuffle_down(sum_sq, offset);
    }

    threadgroup float rms_inv = 0;
    if (threadIdx_x == 0)
        rms_inv = rsqrt(sum_sq / (float)n + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: normalize — read from residual (which now has the sum in FP16)
    // Note: we read back from FP16, so there IS one truncation. But the RMS
    // was computed from the FP32 sum, which is the key improvement.
    for (int i = threadIdx_x; i < n; i += blockDim_x)
    {
        float v = float(res_row[i]);
        float w = float(weight[i]);
        out_row[i] = half(v * rms_inv * w);
    }
}

