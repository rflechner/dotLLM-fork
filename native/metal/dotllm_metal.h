#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dotllm_metal_context dotllm_metal_context;

/// Creates a Metal context (device + command queue + pipeline cache).
/// Returns NULL on failure.
dotllm_metal_context* dotllm_metal_create_context(void);

/// Destroys the context and releases all Metal resources.
void dotllm_metal_destroy_context(dotllm_metal_context* ctx);

/// Element-wise addition: result[i] = a[i] + b[i]
int dotllm_metal_add_f32(
    dotllm_metal_context* ctx,
    const float* a,
    const float* b,
    float* result,
    uint32_t length);

/// Element-wise multiplication: result[i] = a[i] * b[i]
int dotllm_metal_multiply_f32(
    dotllm_metal_context* ctx,
    const float* a,
    const float* b,
    float* result,
    uint32_t length);

int dotllm_metal_softmax_f32(
    dotllm_metal_context* ctx,
    const float* input,
    float* result,
    uint32_t length);



#ifdef __cplusplus
}
#endif
