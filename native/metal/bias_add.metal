#include <metal_stdlib>
using namespace metal;

// Bias addition: output[t, i] += bias[i]  for t in [0, seqLen), i in [0, dim)
// In-place: output is read and written in the same buffer.
// Mirrors the CUDA bias_add_f32 kernel.
kernel void bias_add(
    device float* output         [[ buffer(0) ]],
    device const float* bias     [[ buffer(1) ]],
    constant uint& dim           [[ buffer(2) ]],
    constant uint& seq_len       [[ buffer(3) ]],
    uint id                      [[ thread_position_in_grid ]])
{
    if (id >= dim * seq_len) return;
    output[id] += bias[id % dim];
}
