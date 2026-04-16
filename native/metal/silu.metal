//
//  silu.metal
//  dotllm_metal
//
//  Created by codeanythingpossible on 16.04.2026.
//

#include <metal_stdlib>
using namespace metal;

kernel void silu(
     device const float* input [[buffer(0)]],
     device float* result [[buffer(2)]],
     constant uint& length [[buffer(3)]],
     uint id [[thread_position_in_grid]])
{
    if (id >= length)
    {
        return;
    }

    float x = input[id];
    result[id] = x / (1.0f + exp(-x));
}
