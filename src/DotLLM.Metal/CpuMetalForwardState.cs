using System.Numerics;
using System.Runtime.InteropServices;

namespace DotLLM.Metal;

/// <summary>
/// CPU-allocated forward state — backed by <see cref="NativeMemory.AlignedAlloc"/>.
/// Each kernel call must copy data into a fresh MTLBuffer (input) and memcpy the
/// result back (output). Used as the correctness baseline and when GPU-resident
/// state is not available.
/// </summary>
public sealed class CpuMetalForwardState : IMetalForwardState
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _intermediateSize;
    private readonly int _vocabSize;

    private int _currentSeqLen;

    /// <inheritdoc/>
    public long AllocatedBytes { get; private set; }

    /// <inheritdoc/>
    public nint HiddenState   { get; private set; }
    /// <inheritdoc/>
    public nint Residual      { get; private set; }
    /// <inheritdoc/>
    public nint NormOutput    { get; private set; }
    /// <inheritdoc/>
    public nint Q             { get; private set; }
    /// <inheritdoc/>
    public nint K             { get; private set; }
    /// <inheritdoc/>
    public nint V             { get; private set; }
    /// <inheritdoc/>
    public nint AttnOutput    { get; private set; }
    /// <inheritdoc/>
    public nint FfnGate       { get; private set; }
    /// <inheritdoc/>
    public nint FfnUp         { get; private set; }
    /// <inheritdoc/>
    public nint SiluOutput    { get; private set; }
    /// <inheritdoc/>
    public nint LogitsF16     { get; private set; }
    /// <inheritdoc/>
    public nint LogitsF32     { get; private set; }
    /// <inheritdoc/>
    public nint GemmOutputF16 { get; private set; }
    /// <inheritdoc/>
    public nint DequantScratch { get; private set; }
    /// <inheritdoc/>
    public nint TokenIds      { get; private set; }
    /// <inheritdoc/>
    public nint Positions     { get; private set; }

    /// <summary>
    /// Allocates fixed-size buffers (logits, dequant scratch) and an initial
    /// set of seq-length-dependent buffers sized for decode (<c>seqLen = 1</c>).
    /// </summary>
    public CpuMetalForwardState(int hiddenSize, int numHeads, int numKvHeads, int headDim,
                                 int intermediateSize, int vocabSize)
    {
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _intermediateSize = intermediateSize;
        _vocabSize = vocabSize;
        _currentSeqLen = 0;

        LogitsF16 = AllocDevice((long)vocabSize * sizeof(ushort));
        LogitsF32 = AllocDevice((long)vocabSize * sizeof(float));

        long maxProjectionElements = Math.Max(
            (long)Math.Max(numHeads * headDim, numKvHeads * headDim) * hiddenSize,
            (long)intermediateSize * hiddenSize);
        DequantScratch = AllocDevice(maxProjectionElements * sizeof(ushort));

        EnsureCapacity(1);
    }

    /// <inheritdoc/>
    public void EnsureCapacity(int seqLen)
    {
        if (seqLen <= _currentSeqLen)
            return;

        int newCapacity = (int)BitOperations.RoundUpToPowerOf2((uint)seqLen);
        FreeSequenceBuffers();

        const int half = sizeof(ushort);

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
        nint p;
        p = HiddenState;   FreeIfNonZero(ref p); HiddenState   = p;
        p = Residual;      FreeIfNonZero(ref p); Residual      = p;
        p = NormOutput;    FreeIfNonZero(ref p); NormOutput    = p;
        p = Q;             FreeIfNonZero(ref p); Q             = p;
        p = K;             FreeIfNonZero(ref p); K             = p;
        p = V;             FreeIfNonZero(ref p); V             = p;
        p = AttnOutput;    FreeIfNonZero(ref p); AttnOutput    = p;
        p = FfnGate;       FreeIfNonZero(ref p); FfnGate       = p;
        p = FfnUp;         FreeIfNonZero(ref p); FfnUp         = p;
        p = SiluOutput;    FreeIfNonZero(ref p); SiluOutput    = p;
        p = GemmOutputF16; FreeIfNonZero(ref p); GemmOutputF16 = p;
        p = TokenIds;      FreeIfNonZero(ref p); TokenIds      = p;
        p = Positions;     FreeIfNonZero(ref p); Positions     = p;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        FreeSequenceBuffers();
        nint p;
        p = LogitsF16;      FreeIfNonZero(ref p); LogitsF16      = p;
        p = LogitsF32;      FreeIfNonZero(ref p); LogitsF32      = p;
        p = DequantScratch; FreeIfNonZero(ref p); DequantScratch = p;
        _currentSeqLen = 0;
    }
}
