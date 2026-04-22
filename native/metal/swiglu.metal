// Fused SwiGLU activation kernels for dotLLM.
// out[i] = SiLU(gate[i]) * up[i]  where SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
// ISO port of swiglu_f32.cu (swiglu_f32) and swiglu.cu (swiglu_f16).

#include <metal_stdlib>
using namespace metal;

// ── swiglu_f32 ────────────────────────────────────────────────────────────────
// Port of swiglu_f32.cu::swiglu_f32
kernel void swiglu_f32(
    device const float* gate   [[buffer(0)]],
    device const float* up     [[buffer(1)]],
    device       float* output [[buffer(2)]],
    constant uint&      length [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= length) return;

    float g = gate[idx];
    output[idx] = (g / (1.0f + exp(-g))) * up[idx];
}

// ── swiglu_f16 ────────────────────────────────────────────────────────────────
// Vectorized: half2 loads/stores, FP32 computation for sigmoid precision.
// Port of swiglu.cu::swiglu_f16
kernel void swiglu_f16(
    device const half* gate   [[buffer(0)]],
    device const half* up     [[buffer(1)]],
    device       half* output [[buffer(2)]],
    constant uint&     length [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    uint total2 = length / 2u;

    if (idx < total2)
    {
        device const half2* gate2 = (device const half2*)gate;
        device const half2* up2   = (device const half2*)up;
        device       half2* out2  = (device half2*)output;

        half2 g2 = gate2[idx];
        half2 u2 = up2[idx];

        float g0 = float(g2.x), g1 = float(g2.y);
        float u0 = float(u2.x), u1 = float(u2.y);

        float s0 = (g0 / (1.0f + exp(-g0))) * u0;
        float s1 = (g1 / (1.0f + exp(-g1))) * u1;

        out2[idx] = half2(half(s0), half(s1));
    }

    // Handle odd trailing element
    if ((length & 1u) && idx == 0u)
    {
        uint last = length - 1u;
        float g = float(gate[last]);
        output[last] = half((g / (1.0f + exp(-g))) * float(up[last]));
    }
}
