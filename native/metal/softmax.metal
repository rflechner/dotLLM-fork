#include <metal_stdlib>
using namespace metal;

kernel void softmax(
    device const float* input [[buffer(0)]],
    device float* result [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= length)
    {
        return;
    }
    
    // Step 1: find the max value
    float max_value = -INFINITY;
    for (uint i=0; i<length; i++)
        max_value = max(input[i], max_value);
    
    // Step 2: compute sum of exp(x - max)
    float sum = 0.0f;
    for (uint i = 0; i < length; i++)
        sum += exp(input[i] - max_value);

    // Passe 3 : normalize
    for (uint i = 0; i < length; i++)
        result[i] = exp(input[i] - max_value) / sum;
}
