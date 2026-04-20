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

/// RMS Normalization: output[t, i] = input[t, i] / rms(input[t]) * weight[i]
/// One threadgroup per token — reduction kernel, NOT element-wise.
int dotllm_metal_rmsnorm_f32(
    dotllm_metal_context* ctx,
    const float* input,
    const float* weight,
    float*       output,
    int32_t      n,
    int32_t      seq_len,
    float        eps);

/// Fused residual-add + RMS normalization (FP16).
/// Pass 1: sum = FP32(residual[i]) + FP32(x[i]); residual[i] = FP16(sum); accumulate sum².
/// Pass 2: output[i] = FP16(FP32(residual[i]) * rms_inv * FP32(weight[i])).
/// One threadgroup per token.
int dotllm_metal_fused_add_rmsnorm_f16(
    dotllm_metal_context* ctx,
    uint16_t*       residual,
    const uint16_t* x,
    const uint16_t* weight,
    uint16_t*       output,
    int32_t         n,
    int32_t         seq_len,
    float           eps);

/// Per-head RMS Normalization (in-place): for each (token, head), normalizes head_dim elements.
/// Used by models with QK-norm (Gemma 2, Cohere). One threadgroup per (token × head).
int dotllm_metal_per_head_rmsnorm_f32(
    dotllm_metal_context* ctx,
    float*       qk,
    const float* weight,
    int32_t      num_heads,
    int32_t      head_dim,
    int32_t      seq_len,
    float        eps);

/// Embedding lookup — FP32 table → FP32 output.
/// output[t] = embed_table[token_ids[t]]  (hidden_size floats copied per token).
int dotllm_metal_embedding_f32_f32out(
    dotllm_metal_context* ctx,
    const float* embed_table,
    const int32_t* token_ids,
    float*         output,
    int32_t        vocab_size,
    int32_t        hidden_size,
    int32_t        seq_len);

/// Embedding lookup — FP16 table → FP32 output (cast on the fly).
int dotllm_metal_embedding_f16_f32out(
    dotllm_metal_context* ctx,
    const uint16_t* embed_table,
    const int32_t*  token_ids,
    float*          output,
    int32_t         vocab_size,
    int32_t         hidden_size,
    int32_t         seq_len);

/// Embedding lookup — Q8_0 quantized table → FP32 output (dequantize on the fly).
/// embed_table layout: blocks of Q8_0_BLOCK_BYTES (2-byte half scale + 32 int8 values).
int dotllm_metal_embedding_q8_0_f32out(
    dotllm_metal_context* ctx,
    const uint8_t* embed_table,
    const int32_t* token_ids,
    float*         output,
    int32_t        vocab_size,
    int32_t        hidden_size,
    int32_t        seq_len);

// ── KV-cache quantization ────────────────────────────────────────────────────

/// FP16 → Q8_0 quantization (KV-cache eviction).
/// src: total_blocks × 32 half values.
/// dst: total_blocks × 34 bytes (2-byte half scale + 32 int8 values).
int dotllm_metal_quant_f16_to_q8_0(
    dotllm_metal_context* ctx,
    const uint16_t* src,
    uint8_t*        dst,
    int32_t         total_blocks);

/// FP16 → Q4_0 quantization (KV-cache eviction).
/// src: total_blocks × 32 half values.
/// dst: total_blocks × 18 bytes (2-byte half scale + 16 packed nibble bytes).
int dotllm_metal_quant_f16_to_q4_0(
    dotllm_metal_context* ctx,
    const uint16_t* src,
    uint8_t*        dst,
    int32_t         total_blocks);

// ── Quantized GEMV ───────────────────────────────────────────────────────────

/// Quantized GEMV: y[i] = dot(dequant(W_q8_0[i,:]), x)  for i in [0, n).
/// weight layout: n rows × (k/32) blocks × 34 bytes per block.
/// x: float32 input vector, length k. y: float32 output vector, length n.
/// k must be a multiple of 32.
int dotllm_metal_quantized_gemv_q8_0_f32in(
    dotllm_metal_context* ctx,
    const uint8_t* weight,
    const float*   x,
    float*         y,
    int32_t        n,
    int32_t        k);

// ── Dequantization ───────────────────────────────────────────────────────────

/// Dequantize Q8_0 → FP16.
/// src layout: total_blocks × 34 bytes (2-byte half scale + 32 int8 weights).
/// dst layout: total_blocks × 32 halves.
int dotllm_metal_dequant_q8_0_f16(
    dotllm_metal_context* ctx,
    const uint8_t* src,
    uint16_t*      dst,
    int32_t        total_blocks);

/// Dequantize Q4_0 → FP16.
/// src layout: total_blocks × 18 bytes (2-byte half scale + 16 packed nibble bytes).
/// dst layout: total_blocks × 32 halves.
int dotllm_metal_dequant_q4_0_f16(
    dotllm_metal_context* ctx,
    const uint8_t* src,
    uint16_t*      dst,
    int32_t        total_blocks);

/// Dequantize Q5_0 → FP16.
/// src layout: total_blocks × 22 bytes (2-byte half scale + 4-byte qh mask + 16 packed nibble bytes).
/// dst layout: total_blocks × 32 halves.
int dotllm_metal_dequant_q5_0_f16(
    dotllm_metal_context* ctx,
    const uint8_t* src,
    uint16_t*      dst,
    int32_t        total_blocks);

/// Dequantize Q4_K → FP16.
/// src layout: total_superblocks × 144 bytes.
/// dst layout: total_superblocks × 256 halves.
int dotllm_metal_dequant_q4_k_f16(
    dotllm_metal_context* ctx,
    const uint8_t* src,
    uint16_t*      dst,
    int32_t        total_superblocks);

/// Dequantize Q5_K → FP16.
/// src layout: total_superblocks × 176 bytes.
/// dst layout: total_superblocks × 256 halves.
int dotllm_metal_dequant_q5_k_f16(
    dotllm_metal_context* ctx,
    const uint8_t* src,
    uint16_t*      dst,
    int32_t        total_superblocks);

/// Dequantize Q6_K → FP16.
/// src layout: total_superblocks × 210 bytes.
/// dst layout: total_superblocks × 256 halves.
int dotllm_metal_dequant_q6_k_f16(
    dotllm_metal_context* ctx,
    const uint8_t* src,
    uint16_t*      dst,
    int32_t        total_superblocks);

/// Type conversion: float16 → float32, element-wise.
/// src and dst must each hold n elements (src: n×2 bytes, dst: n×4 bytes).
/// uint16_t* is used for half because C has no standard half type.
int dotllm_metal_convert_f16_to_f32(
    dotllm_metal_context* ctx,
    const uint16_t* src,
    float*          dst,
    int32_t         n);

/// Type conversion: float32 → float16, element-wise.
int dotllm_metal_convert_f32_to_f16(
    dotllm_metal_context* ctx,
    const float* src,
    uint16_t*    dst,
    int32_t      n);

#ifdef __cplusplus
}
#endif
