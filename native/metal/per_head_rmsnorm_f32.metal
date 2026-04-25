#include <metal_stdlib>
using namespace metal;

kernel void per_head_rmsnorm_f32(
    device float* qk            [[ buffer(0) ]],
    device const float* weight  [[ buffer(1) ]],
    constant float& eps         [[ buffer(2) ]],
    constant int& num_heads     [[ buffer(3) ]],
    constant int& head_dim      [[ buffer(4) ]],
    constant int& seq_len       [[ buffer(5) ]],
    
    uint block_id               [[ threadgroup_position_in_grid ]],   // blockIdx.x
    uint threadIdx_x            [[ thread_position_in_threadgroup ]], // threadIdx.x
    uint blockDim_x             [[ threads_per_threadgroup ]],        // blockDim.x
    uint simd_group_width       [[threads_per_simdgroup]])
{
    uint t = block_id / (uint)num_heads;
    uint h = block_id % (uint)num_heads;
    if (t >= (uint)seq_len) return;

    device float* vec = qk + (size_t)t * (size_t)num_heads * head_dim + h * head_dim;
    float sum_sq = 0.0f;
    
    for (int i = threadIdx_x; i < head_dim; i += blockDim_x) {
        float v = vec[i];
        sum_sq += v * v;
    }

    for (int off = simd_group_width/2; off > 0; off >>= 1)
        sum_sq += simd_shuffle_down(sum_sq, off);
    
    threadgroup float ws[32];
    int lane = threadIdx_x % simd_group_width;
    int wid = threadIdx_x / simd_group_width;
    
    if (lane == 0)
        ws[wid] = sum_sq;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (wid == 0) {
        int nw = (blockDim_x + simd_group_width-1)/simd_group_width;
        sum_sq = (lane<nw)?ws[lane]:0.0f;
        
        for (int off=simd_group_width/2; off>0; off>>=1)
            sum_sq += simd_shuffle_down(sum_sq, off);
    }
    
    threadgroup float ri = 0;
    if (threadIdx_x==0)
        ri = rsqrt(sum_sq/(float)head_dim+eps);
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (int i = threadIdx_x; i < head_dim; i += blockDim_x)
        vec[i] = vec[i] * ri * weight[i];
}


// Per-head RMS Normalization (FP16, in-place).
// Port of per_head_rmsnorm_f16.cu. FP16 I/O with FP32 accumulation.
// One threadgroup per (token, head) pair.
kernel void per_head_rmsnorm_f16(
    device       half*  qk        [[ buffer(0) ]],   // in-place [seqLen, num_heads, head_dim]
    device const half*  weight    [[ buffer(1) ]],   // [head_dim]
    constant     float& eps       [[ buffer(2) ]],
    constant     int&   num_heads [[ buffer(3) ]],
    constant     int&   head_dim  [[ buffer(4) ]],
    constant     int&   seq_len   [[ buffer(5) ]],
    uint block_id [[ threadgroup_position_in_grid ]],
    uint lid      [[ thread_position_in_threadgroup ]],
    uint tgSize   [[ threads_per_threadgroup ]])
{
    uint t = block_id / (uint)num_heads;
    uint h = block_id % (uint)num_heads;
    if (t >= (uint)seq_len) return;

    device half* vec = qk + (size_t)t * (size_t)num_heads * head_dim + h * head_dim;

    threadgroup float simd_sums[32];
    threadgroup float ri = 0.0f;

    // Phase 1: partial sum of squares — load half → float
    float sum_sq = 0.0f;
    for (uint i = lid; i < (uint)head_dim; i += tgSize) {
        float v = float(vec[i]);
        sum_sq += v * v;
    }

    // Phase 2: simd-group reduction
    sum_sq = simd_sum(sum_sq);

    // Phase 3: cross-simd reduction via threadgroup memory
    uint simd_lane = lid % 32;
    uint simd_id   = lid / 32;
    if (simd_lane == 0) {
        simd_sums[simd_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: first 32 threads reduce simd_sums[]
    if (lid < 32) {
        uint num_simds = (tgSize + 31) / 32;
        sum_sq = (lid < num_simds) ? simd_sums[lid] : 0.0f;
        sum_sq = simd_sum(sum_sq);
    }

    // Phase 5: thread 0 computes and broadcasts ri
    if (lid == 0) {
        ri = rsqrt(sum_sq / (float)head_dim + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 6: normalize, scale, write back as half
    float scale = ri;
    for (uint i = lid; i < (uint)head_dim; i += tgSize) {
        vec[i] = half(float(vec[i]) * scale * float(weight[i]));
    }
}

