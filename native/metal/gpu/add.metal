// Element-wise addition kernels for dotLLM.
// ISO port of add.cu (add_f16) and add_f32.cu (add_f32, add_f32_f16).

#include <metal_stdlib>
using namespace metal;

// ── add_f32 ──────────────────────────────────────────────────────────────────
// output[i] = a[i] + b[i]  (all FP32, in-place safe)
// Port of add_f32.cu::add_f32
kernel void add_f32(
    device const float* a  [[buffer(0)]],
    device const float* b  [[buffer(1)]],
    device       float* output [[buffer(2)]],
    constant uint&      n  [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx < n)
        output[idx] = a[idx] + b[idx];
}

// ── add_f16 ──────────────────────────────────────────────────────────────────
// output[i] = a[i] + b[i]  (FP16, in-place safe: output may alias a or b)
// Vectorized: half2 packed operations process 2 elements per thread.
// Port of add.cu::add_f16
kernel void add_f16(
    device const half* a      [[buffer(0)]],
    device const half* b      [[buffer(1)]],
    device       half* output [[buffer(2)]],
    constant uint&     n      [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    uint n2 = n / 2u;

    if (idx < n2)
    {
        device const half2* a2   = (device const half2*)a;
        device const half2* b2   = (device const half2*)b;
        device       half2* out2 = (device half2*)output;
        out2[idx] = a2[idx] + b2[idx];
    }

    // Handle odd trailing element (single thread)
    if ((n & 1u) && idx == 0u)
    {
        uint last = n - 1u;
        output[last] = a[last] + b[last];
    }
}

// ── add_f32_f16 ──────────────────────────────────────────────────────────────
// output_f32[i] = a_f32[i] + b_f16[i]
// Used when adding FP16 projection output into FP32 residual stream.
// Port of add_f32.cu::add_f32_f16
kernel void add_f32_f16(
    device const float* a      [[buffer(0)]],
    device const half*  b      [[buffer(1)]],
    device       float* output [[buffer(2)]],
    constant uint&      n      [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx < n)
        output[idx] = a[idx] + float(b[idx]);
}
