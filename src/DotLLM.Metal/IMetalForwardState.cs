namespace DotLLM.Metal;

/// <summary>
/// Pre-allocated scratch buffers for the Metal forward pass.
///
/// Two implementations exist:
/// <list type="bullet">
///   <item><description>
///     <see cref="CpuMetalForwardState"/> — allocates via <c>NativeMemory.AlignedAlloc</c>.
///     Each kernel call must copy CPU memory into a fresh MTLBuffer and copy the
///     result back. Kept for correctness baseline and CPU-side debugging.
///   </description></item>
///   <item><description>
///     <c>GpuMetalForwardState</c> (planned) — allocates as <c>MTLResourceStorageModeShared</c>
///     MTLBuffers via Objective-C interop. The pointers exposed here are
///     <c>MTLBuffer.contents</c>, so kernels can wrap them zero-copy and the same
///     bytes are visible to both CPU and GPU. Required for batched-command-buffer
///     execution.
///   </description></item>
/// </list>
///
/// All activation buffers are FP16 (2 bytes / element). Logits are kept in both
/// FP16 (kernel output) and FP32 (sampler input).
/// </summary>
public interface IMetalForwardState : IDisposable
{
    /// <summary>Total bytes currently allocated (sum of all live buffers).</summary>
    long AllocatedBytes { get; }

    /// <summary>Hidden state, shape <c>[seqLen, hiddenSize]</c>.</summary>
    nint HiddenState { get; }

    /// <summary>Residual stream, shape <c>[seqLen, hiddenSize]</c>.</summary>
    nint Residual { get; }

    /// <summary>Output of the pre-attention RMSNorm, shape <c>[seqLen, hiddenSize]</c>.</summary>
    nint NormOutput { get; }

    /// <summary>Query projection, shape <c>[seqLen, numHeads * headDim]</c>.</summary>
    nint Q { get; }

    /// <summary>Key projection, shape <c>[seqLen, numKvHeads * headDim]</c>.</summary>
    nint K { get; }

    /// <summary>Value projection, shape <c>[seqLen, numKvHeads * headDim]</c>.</summary>
    nint V { get; }

    /// <summary>Attention output, shape <c>[seqLen, numHeads * headDim]</c>.</summary>
    nint AttnOutput { get; }

    /// <summary>SwiGLU gate projection, shape <c>[seqLen, intermediateSize]</c>.</summary>
    nint FfnGate { get; }

    /// <summary>SwiGLU up projection, shape <c>[seqLen, intermediateSize]</c>.</summary>
    nint FfnUp { get; }

    /// <summary>Output of SiLU(gate) · up, shape <c>[seqLen, intermediateSize]</c>.</summary>
    nint SiluOutput { get; }

    /// <summary>Logits in FP16 (kernel output), shape <c>[vocabSize]</c>.</summary>
    nint LogitsF16 { get; }

    /// <summary>Logits in FP32 (sampler input), shape <c>[vocabSize]</c>.</summary>
    nint LogitsF32 { get; }

    /// <summary>General-purpose FP16 scratch sized for the largest projection or LM head.</summary>
    nint GemmOutputF16 { get; }

    /// <summary>FP16 scratch buffer for on-the-fly dequantization of quantized weights.</summary>
    nint DequantScratch { get; }

    /// <summary>Token IDs for the current step, shape <c>[seqLen]</c> int32.</summary>
    nint TokenIds { get; }

    /// <summary>Position indices for the current step, shape <c>[seqLen]</c> int32.</summary>
    nint Positions { get; }

    /// <summary>
    /// Ensures all per-token scratch buffers are large enough for
    /// <paramref name="seqLen"/> tokens. Implementations should grow geometrically
    /// to amortize reallocation cost across consecutive prefills.
    /// </summary>
    void EnsureCapacity(int seqLen);
}
