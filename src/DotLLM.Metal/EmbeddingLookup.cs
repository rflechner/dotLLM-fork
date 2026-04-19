using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Metal.Interop;

namespace DotLLM.Metal;

/// <summary>
/// Embedding lookup kernels accelerated via Metal GPU.
/// Direct translation of embedding_f32out.cu.
/// For each token t: output[t] = dequant(embedTable[tokenIds[t]]) as FP32.
/// </summary>
public static class EmbeddingLookup
{
    private const int s_q80BlockSize  = 32;
    private const int s_q80BlockBytes = 34; // 2-byte half scale + 32 int8 values

    /// <summary>
    /// Embedding lookup from a float32 table.
    /// </summary>
    /// <param name="ctx">The Metal context.</param>
    /// <param name="embedTable">Full embedding table, shape <c>[vocabSize × hiddenSize]</c>.</param>
    /// <param name="tokenIds">Token indices, length <c>seqLen</c>.</param>
    /// <param name="output">Output buffer, shape <c>[seqLen × hiddenSize]</c>.</param>
    /// <param name="vocabSize">Number of vocabulary entries.</param>
    /// <param name="hiddenSize">Embedding dimension.</param>
    /// <param name="seqLen">Number of tokens to look up.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void F32ToF32(
        MetalContext        ctx,
        ReadOnlySpan<float> embedTable,
        ReadOnlySpan<int>   tokenIds,
        Span<float>         output,
        int vocabSize, int hiddenSize, int seqLen)
    {
        Validate(embedTable.Length, vocabSize, hiddenSize, seqLen, sizeof(float), tokenIds, output);

        unsafe
        {
            fixed (float* pTable  = embedTable)
            fixed (int*   pIds    = tokenIds)
            fixed (float* pOut    = output)
            {
                int code = MetalNative.EmbeddingF32F32Out(
                    ctx.Handle, pTable, pIds, pOut, vocabSize, hiddenSize, seqLen);
                if (code != 0)
                    throw new InvalidOperationException(
                        $"Metal embedding_f32_f32out failed with code {code}.");
            }
        }
    }

    /// <summary>
    /// Embedding lookup from a float16 table — cast to float32 on the GPU.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void F16ToF32(
        MetalContext          ctx,
        ReadOnlySpan<Half>    embedTable,
        ReadOnlySpan<int>     tokenIds,
        Span<float>           output,
        int vocabSize, int hiddenSize, int seqLen)
    {
        Validate(embedTable.Length, vocabSize, hiddenSize, seqLen, sizeof(ushort), tokenIds, output);

        ReadOnlySpan<ushort> raw = MemoryMarshal.Cast<Half, ushort>(embedTable);

        unsafe
        {
            fixed (ushort* pTable = raw)
            fixed (int*    pIds   = tokenIds)
            fixed (float*  pOut   = output)
            {
                int code = MetalNative.EmbeddingF16F32Out(
                    ctx.Handle, pTable, pIds, pOut, vocabSize, hiddenSize, seqLen);
                if (code != 0)
                    throw new InvalidOperationException(
                        $"Metal embedding_f16_f32out failed with code {code}.");
            }
        }
    }

    /// <summary>
    /// Embedding lookup from a Q8_0 quantized table — dequantize to float32 on the GPU.
    /// Each block is 34 bytes: 2-byte half scale followed by 32 int8 weights.
    /// </summary>
    /// <param name="ctx">Metal context</param>
    /// <param name="embedTable">Raw Q8_0 bytes, length = <c>vocabSize × (hiddenSize/32) × 34</c>.</param>
    /// <param name="tokenIds">Token ids</param>
    /// <param name="output">Output float32 embeddings</param>
    /// <param name="vocabSize">Vocab size</param>
    /// <param name="hiddenSize">Hidden size</param>
    /// <param name="seqLen">Sequence length</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Q8_0ToF32(
        MetalContext       ctx,
        ReadOnlySpan<byte> embedTable,
        ReadOnlySpan<int>  tokenIds,
        Span<float>        output,
        int vocabSize, int hiddenSize, int seqLen)
    {
        if (hiddenSize % s_q80BlockSize != 0)
            throw new ArgumentException(
                $"hiddenSize ({hiddenSize}) must be a multiple of {s_q80BlockSize} for Q8_0.", nameof(hiddenSize));

        int blocksPerRow      = hiddenSize / s_q80BlockSize;
        int expectedTableLen  = vocabSize * blocksPerRow * s_q80BlockBytes;
        if (embedTable.Length != expectedTableLen)
            throw new ArgumentException(
                $"embedTable.Length ({embedTable.Length}) must equal vocabSize × blocksPerRow × {s_q80BlockBytes} ({expectedTableLen}).", nameof(embedTable));
        if (tokenIds.Length != seqLen)
            throw new ArgumentException(
                $"tokenIds.Length ({tokenIds.Length}) must equal seqLen ({seqLen}).", nameof(tokenIds));
        if (output.Length != seqLen * hiddenSize)
            throw new ArgumentException(
                $"output.Length ({output.Length}) must equal seqLen × hiddenSize ({seqLen * hiddenSize}).", nameof(output));

        unsafe
        {
            fixed (byte*  pTable = embedTable)
            fixed (int*   pIds   = tokenIds)
            fixed (float* pOut   = output)
            {
                int code = MetalNative.EmbeddingQ8_0F32Out(
                    ctx.Handle, pTable, pIds, pOut, vocabSize, hiddenSize, seqLen);
                if (code != 0)
                    throw new InvalidOperationException(
                        $"Metal embedding_q8_0_f32out failed with code {code}.");
            }
        }
    }

    private static void Validate(
        int tableLen, int vocabSize, int hiddenSize, int seqLen,
        int elemBytes, ReadOnlySpan<int> tokenIds, Span<float> output)
    {
        if (tableLen != vocabSize * hiddenSize)
            throw new ArgumentException(
                $"embedTable.Length ({tableLen}) must equal vocabSize × hiddenSize ({vocabSize * hiddenSize}).");
        if (tokenIds.Length != seqLen)
            throw new ArgumentException(
                $"tokenIds.Length ({tokenIds.Length}) must equal seqLen ({seqLen}).");
        if (output.Length != seqLen * hiddenSize)
            throw new ArgumentException(
                $"output.Length ({output.Length}) must equal seqLen × hiddenSize ({seqLen * hiddenSize}).");
    }
}
