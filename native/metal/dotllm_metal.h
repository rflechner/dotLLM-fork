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

/// SiLU activation: result[i] = input[i] * sigmoid(input[i])
int dotllm_metal_silu_f32(
    dotllm_metal_context* ctx,
    const float* input,
    float* result,
    uint32_t length);

/// SwiGLU: result[i] = gate[i] * sigmoid(gate[i]) * up[i]
int dotllm_metal_swiglu_f32(
    dotllm_metal_context* ctx,
    const float* gate,
    const float* up,
    float* result,
    uint32_t length);

/// Rotary Position Embedding (in-place). Direct translation of rope_f32.cu.
/// rope_type: 0 = norm/interleaved (Llama/Mistral), 1 = neox/split (Qwen/Phi).
int dotllm_metal_rope_f32(
    dotllm_metal_context* ctx,
    float*         q,
    float*         k,
    const int32_t* positions,
    int32_t        seq_len,
    int32_t        num_heads,
    int32_t        num_kv_heads,
    int32_t        head_dim,
    int32_t        rope_dim,
    float          theta,
    int32_t        rope_type);

/// Bias addition (in-place): output[t, i] += bias[i]  for t in [0, seq_len), i in [0, dim)
int dotllm_metal_bias_add_f32(
    dotllm_metal_context* ctx,
    float* output,
    const float* bias,
    uint32_t dim,
    uint32_t seq_len);

#ifdef __cplusplus
}
#endif
