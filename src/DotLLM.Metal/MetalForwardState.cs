using System.Numerics;
using System.Runtime.InteropServices;

namespace DotLLM.Metal;

/// <summary>
/// Pre-allocated scratch buffers for the Metal forward pass. Mirrors
/// <c>CudaForwardState</c> field-by-field; the only structural difference
/// is the allocator — <see cref="NativeMemory.AlignedAlloc"/> instead of
/// <c>cuMemAlloc_v2</c>, since Apple Silicon's unified memory makes any
/// aligned heap allocation directly GPU-addressable.
///
/// All activation buffers are FP16 (2 bytes / element). Logits are kept
/// in both FP16 (kernel output) and FP32 (sampler input).
/// </summary>
internal sealed class MetalForwardState : IDisposable
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _intermediateSize;
    private readonly int _vocabSize;

    private int _currentSeqLen;

    /// <summary>Total bytes currently allocated (sum of all live buffers).</summary>
    public long AllocatedBytes { get; private set; }

    // ── Activation buffers (FP16 in unified RAM) ──────────────────────────────

    /// <summary>Hidden state, shape <c>[seqLen, hiddenSize]</c>.</summary>
    public nint HiddenState;

    /// <summary>Residual stream, shape <c>[seqLen, hiddenSize]</c>.</summary>
    public nint Residual;

    /// <summary>Output of the pre-attention RMSNorm, shape <c>[seqLen, hiddenSize]</c>.</summary>
    public nint NormOutput;

    /// <summary>Query projection, shape <c>[seqLen, numHeads * headDim]</c>.</summary>
    public nint Q;

    /// <summary>Key projection, shape <c>[seqLen, numKvHeads * headDim]</c>.</summary>
    public nint K;

    /// <summary>Value projection, shape <c>[seqLen, numKvHeads * headDim]</c>.</summary>
    public nint V;

    /// <summary>Attention output, shape <c>[seqLen, numHeads * headDim]</c>.</summary>
    public nint AttnOutput;

    /// <summary>SwiGLU gate projection, shape <c>[seqLen, intermediateSize]</c>.</summary>
    public nint FfnGate;

    /// <summary>SwiGLU up projection, shape <c>[seqLen, intermediateSize]</c>.</summary>
    public nint FfnUp;

    /// <summary>Output of SiLU(gate) · up, shape <c>[seqLen, intermediateSize]</c>.</summary>
    public nint SiluOutput;

    // ── Logits ────────────────────────────────────────────────────────────────

    /// <summary>Logits in FP16 (kernel output), shape <c>[vocabSize]</c>.</summary>
    public nint LogitsF16;

    /// <summary>Logits in FP32 (sampler input), shape <c>[vocabSize]</c>.</summary>
    public nint LogitsF32;

    // ── Generic scratch ───────────────────────────────────────────────────────

    /// <summary>General-purpose FP16 scratch sized for the largest projection or LM head.</summary>
    public nint GemmOutputF16;

    /// <summary>
    /// FP16 scratch buffer for on-the-fly dequantization of quantized weights
    /// before an MPS GEMM call. Used only by <c>MmapOnlyStrategy</c> during
    /// prefill; ignored when weights are already FP16.
    /// </summary>
    public nint DequantScratch;

    // ── Per-token inputs (unified RAM — no H2D copy) ──────────────────────────

    /// <summary>Token IDs for the current step, shape <c>[seqLen]</c> int32.</summary>
    public nint TokenIds;

    /// <summary>Position indices for the current step, shape <c>[seqLen]</c> int32.</summary>
    public nint Positions;

    /// <summary>
    /// Allocates the fixed-size buffers (logits, dequant scratch) and an
    /// initial set of sequence-length-dependent buffers sized for decode
    /// (<c>seqLen = 1</c>). Call <see cref="EnsureCapacity(int)"/> before
    /// prefill to grow the per-token buffers.
    /// </summary>
    public MetalForwardState(int hiddenSize, int numHeads, int numKvHeads, int headDim,
                              int intermediateSize, int vocabSize)
    {
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _intermediateSize = intermediateSize;
        _vocabSize = vocabSize;
        _currentSeqLen = 0;

        // Logits are fixed-size (only last token is sampled).
        LogitsF16 = AllocDevice((long)vocabSize * sizeof(ushort));
        LogitsF32 = AllocDevice((long)vocabSize * sizeof(float));

        // Dequant scratch: sized for the largest per-layer projection in FP16.
        // Used by MmapOnlyStrategy when prefill goes through MPS GEMM, which
        // requires FP16 inputs.
        long maxProjectionElements = Math.Max(
            (long)Math.Max(numHeads * headDim, numKvHeads * headDim) * hiddenSize,
            (long)intermediateSize * hiddenSize);
        DequantScratch = AllocDevice(maxProjectionElements * sizeof(ushort));

        // Initial allocation for decode (seqLen=1).
        EnsureCapacity(1);
    }

    /// <summary>
    /// Ensures all per-token scratch buffers are large enough for
    /// <paramref name="seqLen"/> tokens. Uses power-of-2 growth to amortize
    /// reallocation cost across consecutive prefills.
    /// </summary>
    public void EnsureCapacity(int seqLen)
    {
        if (seqLen <= _currentSeqLen)
            return;

        int newCapacity = (int)BitOperations.RoundUpToPowerOf2((uint)seqLen);
        FreeSequenceBuffers();

        const int half = sizeof(ushort); // FP16 = 2 bytes

        // All activation buffers are FP16 — per GPU.md spec, memory-bandwidth-optimal.
        // Only LogitsF32 (sampler input) stays FP32.
        HiddenState = AllocDevice((long)newCapacity * _hiddenSize * half);
        Residual    = AllocDevice((long)newCapacity * _hiddenSize * half);
        NormOutput  = AllocDevice((long)newCapacity * _hiddenSize * half);
        Q           = AllocDevice((long)newCapacity * _numHeads   * _headDim * half);
        K           = AllocDevice((long)newCapacity * _numKvHeads * _headDim * half);
        V           = AllocDevice((long)newCapacity * _numKvHeads * _headDim * half);
        AttnOutput  = AllocDevice((long)newCapacity * _numHeads   * _headDim * half);
        FfnGate     = AllocDevice((long)newCapacity * _intermediateSize * half);
        FfnUp       = AllocDevice((long)newCapacity * _intermediateSize * half);
        SiluOutput  = AllocDevice((long)newCapacity * _intermediateSize * half);

        // General scratch must fit the largest projection output OR the LM-head logits.
        long maxPerLayer = (long)newCapacity * Math.Max(
            Math.Max(_numHeads * _headDim, _intermediateSize), _hiddenSize);
        long maxLmHead = _vocabSize;
        GemmOutputF16 = AllocDevice(Math.Max(maxPerLayer, maxLmHead) * half);

        TokenIds  = AllocDevice((long)newCapacity * sizeof(int));
        Positions = AllocDevice((long)newCapacity * sizeof(int));

        _currentSeqLen = newCapacity;
    }

    private unsafe nint AllocDevice(long bytes)
    {
        nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)bytes, alignment: 64);
        AllocatedBytes += bytes;
        return ptr;
    }

    private static unsafe void FreeIfNonZero(ref nint ptr)
    {
        if (ptr == 0) return;
        NativeMemory.AlignedFree((void*)ptr);
        ptr = 0;
    }

    private void FreeSequenceBuffers()
    {
        FreeIfNonZero(ref HiddenState);
        FreeIfNonZero(ref Residual);
        FreeIfNonZero(ref NormOutput);
        FreeIfNonZero(ref Q);
        FreeIfNonZero(ref K);
        FreeIfNonZero(ref V);
        FreeIfNonZero(ref AttnOutput);
        FreeIfNonZero(ref FfnGate);
        FreeIfNonZero(ref FfnUp);
        FreeIfNonZero(ref SiluOutput);
        FreeIfNonZero(ref GemmOutputF16);
        FreeIfNonZero(ref TokenIds);
        FreeIfNonZero(ref Positions);
    }

    /// <summary>Frees every allocated buffer. Safe to call multiple times.</summary>
    public void Dispose()
    {
        FreeSequenceBuffers();
        FreeIfNonZero(ref LogitsF16);
        FreeIfNonZero(ref LogitsF32);
        FreeIfNonZero(ref DequantScratch);
        _currentSeqLen = 0;
    }
}
