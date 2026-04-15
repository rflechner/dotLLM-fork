#include <metal_stdlib>
using namespace metal;

kernel void add_arrays(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= length)
    {
        return;
    }

    result[id] = a[id] + b[id];
}
