
#include <metal_stdlib>
using namespace metal;

kernel void swiglu(
    device const float* gate   [[buffer(0)]],  // moitié SiLU
    device const float* up     [[buffer(1)]],  // moitié linéaire
    device float*       result [[buffer(2)]],
    constant uint&      length [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= length) return;

    float g = gate[id];
    float silu = g / (1.0f + exp(-g));
    result[id] = silu * up[id];
}


