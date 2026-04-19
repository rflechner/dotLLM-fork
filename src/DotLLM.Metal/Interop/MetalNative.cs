using System.Runtime.InteropServices;

namespace DotLLM.Metal.Interop;

/// <summary>
/// P/Invoke declarations for the native Metal library (libdotllmmetal.dylib).
/// All kernel functions take an opaque context handle as first argument.
/// </summary>
internal static partial class MetalNative
{
    private const string LibName = "dotllmmetal";

    // ── Context ──────────────────────────────────────────────────────────

    /// <summary>Creates a Metal context (device + command queue + pipeline cache).</summary>
    /// <returns>Opaque pointer, or <c>0</c> on failure.</returns>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_create_context")]
    internal static partial nint CreateContext();

    /// <summary>Destroys the context and releases all Metal resources.</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_destroy_context")]
    [SuppressGCTransition]
    internal static partial void DestroyContext(nint ctx);

    // ── Kernels ───────────────────────────────────────────────────────────

    /// <summary>Element-wise addition: result[i] = a[i] + b[i]</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_add_f32")]
    internal static unsafe partial int AddF32(
        nint ctx,
        float* a,
        float* b,
        float* result,
        uint length);

    /// <summary>Element-wise multiplication: result[i] = a[i] * b[i]</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_multiply_f32")]
    internal static unsafe partial int MultiplyF32(
        nint ctx,
        float* a,
        float* b,
        float* result,
        uint length);

    /// <summary>Softmax activation function: result[i] = exp(input[i]) / sum(exp(input))</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_softmax_f32")]
    internal static unsafe partial int SoftmaxF32(
        nint ctx,
        float* input,
        float* result,
        uint length);

    [LibraryImport(LibName, EntryPoint = "dotllm_metal_silu_f32")]
    internal static unsafe partial int SiluF32(
        nint ctx,
        float* input,
        float* result,
        uint length);

    [LibraryImport(LibName, EntryPoint = "dotllm_metal_swiglu_f32")]
    internal static unsafe partial int SwigluF32(
        nint ctx,
        float* gate,
        float* up,
        float* result,
        uint length);

    /// <summary>
    /// Rotary Position Embedding (in-place). Direct translation of rope_f32.cu.
    /// ropeType: 0 = norm/interleaved (Llama/Mistral), 1 = neox/split (Qwen/Phi).
    /// </summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_rope_f32")]
    internal static unsafe partial int RoPEF32(
        nint   ctx,
        float* q,
        float* k,
        int*   positions,
        int    seqLen,
        int    numHeads,
        int    numKvHeads,
        int    headDim,
        int    ropeDim,
        float  theta,
        int    ropeType);

    /// <summary>Bias addition (in-place): output[t, i] += bias[i]</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_bias_add_f32")]
    internal static unsafe partial int BiasAddF32(
        nint ctx,
        float* output,
        float* bias,
        uint dim,
        uint seqLen);

    /// <summary>
    /// RMS Normalization: output[t, i] = input[t, i] / rms(input[t]) * weight[i].
    /// Reduction kernel — one threadgroup per token, NOT element-wise.
    /// </summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_rmsnorm_f32")]
    internal static unsafe partial int RmsNormF32(
        nint   ctx,
        float* input,
        float* weight,
        float* output,
        int    n,
        int    seqLen,
        float  eps);

    /// <summary>
    /// Per-head RMS Normalization (in-place).
    /// For each (token, head), normalizes head_dim elements: vec = vec / rms(vec) * weight.
    /// One threadgroup per (token × head).
    /// </summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_per_head_rmsnorm_f32")]
    internal static unsafe partial int PerHeadRmsNormF32(
        nint   ctx,
        float* qk,
        float* weight,
        int    numHeads,
        int    headDim,
        int    seqLen,
        float  eps);

    /// <summary>
    /// Fused residual-add + RMS normalization (FP16 I/O).
    /// residual is updated in-place (sum written back as f16); output receives the normalized result.
    /// All half buffers passed as ushort* (same bit layout as System.Half).
    /// </summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_fused_add_rmsnorm_f16")]
    internal static unsafe partial int FusedAddRmsNormF16(
        nint    ctx,
        ushort* residual,
        ushort* x,
        ushort* weight,
        ushort* output,
        int     n,
        int     seqLen,
        float   eps);

    /// <summary>Converts n float16 values to float32. src is passed as ushort* (same bit layout as Half).</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_convert_f16_to_f32")]
    internal static unsafe partial int ConvertF16ToF32(
        nint    ctx,
        ushort* src,
        float*  dst,
        int     n);

    /// <summary>Converts n float32 values to float16. dst is passed as ushort* (same bit layout as Half).</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_convert_f32_to_f16")]
    internal static unsafe partial int ConvertF32ToF16(
        nint    ctx,
        float*  src,
        ushort* dst,
        int     n);
}
