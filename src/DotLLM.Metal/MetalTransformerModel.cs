using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Metal.Kernels;
using DotLLM.Metal.Weights;
using DotLLM.Models.Gguf;

namespace DotLLM.Metal;

/// <summary>
/// Metal implementation of <see cref="IModel"/>.
/// </summary>
public sealed unsafe class MetalTransformerModel : IModel
{
    private readonly MetalWeights _weights;
    private readonly MetalForwardState _state;
    private readonly MetalContext _context;
    private readonly GgufFile _gguf;
    private readonly int _deviceId;
    private readonly float _ropeTheta;
    private readonly int _ropeDim;
    private readonly int _ropeType;
    private bool _disposed;

    private MetalTransformerModel(MetalWeights weights, MetalForwardState state, MetalContext context, GgufFile gguf, int deviceId, float ropeTheta, int ropeDim, int ropeType, ModelConfig config)
    {
        _weights = weights;
        _state = state;
        _context = context;
        _gguf = gguf;
        _deviceId = deviceId;
        _ropeTheta = ropeTheta;
        _ropeDim = ropeDim;
        _ropeType = ropeType;
        Config = config;
    }

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _state.AllocatedBytes;

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, null);

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
    {
        int seqLen          = tokenIds.Length;
        int hiddenSize      = Config.HiddenSize;
        int numHeads        = Config.NumAttentionHeads;
        int numKvHeads      = Config.NumKvHeads;
        int headDim         = Config.HeadDim;
        int vocabSize       = Config.VocabSize;
        float eps           = Config.NormEpsilon;

        _state.EnsureCapacity(seqLen);

        // 1. Copy inputs (unified memory → simple copy)
        CopyInputToState(tokenIds, _state.TokenIds);
        CopyInputToState(positions,  _state.Positions);

        // 2. Embedding
        var embedding = _weights.TokenEmbedding;
        nint tablePtr = embedding.Fp16Pointer != 0
            ? embedding.Fp16Pointer       // FP16 si la stratégie l'a produit
            : embedding.QuantizedPointer; // sinon, format quantisé du mmap
        QuantizationType tableFormat = embedding.Fp16Pointer != 0
            ? QuantizationType.F16
            : embedding.QuantizedFormat;
        EmbeddingLookup.Execute(
            _context, tablePtr, tableFormat,
            _state.TokenIds, _state.HiddenState,
            seqLen, Config.HiddenSize, Config.VocabSize);

        // 3. Layer 0 setup
        Memcpy(_state.Residual, _state.HiddenState, seqLen * hiddenSize * 2);
        RmsNormF16.Execute(_context, _state.HiddenState, _weights.Layers[0].AttnNormWeight,
                           _state.NormOutput, hiddenSize, seqLen, eps);

        // 4. Boucle des layers
        for (int layer = 0; layer < Config.NumLayers; layer++)
        {
            ref readonly var lw = ref _weights.Layers[layer];

            // Q/K/V
            Project(lw.Q, _state.NormOutput, _state.Q, seqLen);
            Project(lw.K, _state.NormOutput, _state.K, seqLen);
            Project(lw.V, _state.NormOutput, _state.V, seqLen);

            // Biases optionnels
            if (lw.QBias != 0) BiasAddF16.Execute(_context, _state.Q, lw.QBias, lw.Q.OutputDim, seqLen);
            // … idem K, V

            // QK-norm optionnels
            if (lw.QNormWeight != 0)
                PerHeadRmsNormF16.Execute(_context, _state.Q, lw.QNormWeight, numHeads, headDim, seqLen, eps);
            if (lw.KNormWeight != 0)
                PerHeadRmsNormF16.Execute(_context, _state.K, lw.KNormWeight, numKvHeads, headDim, seqLen, eps);

            // RoPE
            RoPEF16.Execute(_context, _state.Q, _state.K, _state.Positions,
                             numHeads, numKvHeads, headDim,
                            _ropeDim, _ropeTheta, seqLen, (RoPEType)_ropeType);

            // KV cache + Attention (TODO : kvCache plus tard, pour la v1 attention sur K/V courants seulement)
            AttentionF16.Execute(_context, _state.Q, _state.K, _state.V, _state.AttnOutput,
                                 seqLen, seqLen, numHeads, numKvHeads, headDim,
                                 positionOffset: 0, slidingWindow: 0);

            // O projection
            Project(lw.O, _state.AttnOutput, _state.NormOutput, seqLen);

            // Residual + FFN-norm
            FusedAddRmsNorm.Execute(_context, _state.Residual, _state.NormOutput,
                                    lw.FfnNormWeight, _state.NormOutput,
                                    hiddenSize, seqLen, eps);

            // FFN
            Project(lw.Gate, _state.NormOutput, _state.FfnGate, seqLen);
            Project(lw.Up,   _state.NormOutput, _state.FfnUp,   seqLen);
            SwigluF16.Execute(_context, _state.FfnGate, _state.FfnUp, _state.SiluOutput,
                              seqLen * Config.IntermediateSize);
            Project(lw.Down, _state.SiluOutput, _state.NormOutput, seqLen);

            // Residual + norme attn du layer suivant (sauf au dernier)
            if (layer + 1 < Config.NumLayers)
            {
                ref readonly var nextLw = ref _weights.Layers[layer + 1];
                FusedAddRmsNorm.Execute(_context, _state.Residual, _state.NormOutput,
                                        nextLw.AttnNormWeight, _state.NormOutput,
                                        hiddenSize, seqLen, eps);
            }
        }

        // 5. Final RmsNorm — last token only
        nint lastHidden = _state.Residual + (nint)((seqLen - 1) * hiddenSize * 2);
        RmsNormF16.Execute(_context, lastHidden, _weights.OutputNormWeight,
                           _state.NormOutput, hiddenSize, 1, eps);

        // 6. LM head → vocab logits
        Project(_weights.LmHead, _state.NormOutput, _state.LogitsF16, seqLen: 1);

        // 7. F16 → F32 pour le sampler
        EmbeddingLookup.F16ToF32(
            _context,
            _state.LogitsF16,
            _state.TokenIds,
            _state.LogitsF32,
            vocabSize,
            hiddenSize,
            seqLen);

        // 8. Wrap en ITensor (host-readable directement, pas de D2H grâce à la mémoire unifiée)
        return new MetalTensor(_state.LogitsF32, vocabSize);
    }

    // Dans MetalTransformerModel
    private static void CopyInputToState(ReadOnlySpan<int> src, nint dst)
    {
        fixed (int* p = src)
        {
            NativeMemory.Copy(p, (void*)dst, (nuint)(src.Length * sizeof(int)));
        }
    }

    private void Memcpy(IntPtr stateResidual, IntPtr stateHiddenState, int size)
    {
        NativeMemory.Copy(
            source:      (void*)stateHiddenState,
            destination: (void*)stateResidual,
            byteCount:   (nuint)(size * sizeof(ushort)));
    }

    private void Project(in LoadedWeight w, nint input, nint output, int seqLen)
    {
        int n = w.OutputDim;   // sortie
        int k = w.InputDim;    // entrée

        if (seqLen > 1)
        {
            // ── Prefill : matrice × matrice ──
            // Besoin d'inputs FP16. Si la stratégie a fourni Fp16Pointer, direct.
            // Sinon (MmapOnly + format quantisé), dequant dans DequantScratch.
            nint weightFp16 = w.Fp16Pointer;
            if (weightFp16 == 0)
            {
                DequantInto(_state.DequantScratch, w);   // dequant on-the-fly
                weightFp16 = _state.DequantScratch;
            }
            Gemm.ExecuteF16(_context, input, weightFp16, output,
                m: seqLen, n: n, k: k,
                transposeA: false, transposeB: true);
        }
        else if (w.QuantizedPointer != 0 && QuantizedGemv.Supports(w.QuantizedFormat))
        {
            // ── Decode + format quantisé supporté : GEMV quantisé ──
            QuantizedGemv.Dispatch(_context, w.QuantizedFormat, w.QuantizedPointer,
                input, output, n, k);
        }
        else
        {
            // ── Decode FP16 : MPS GEMM avec m=1 ──
            Gemm.ExecuteF16(_context, input, w.Fp16Pointer, output,
                m: 1, n: n, k: k,
                transposeA: false, transposeB: true);
        }
    }

    private void DequantInto(nint destinationFp16, in LoadedWeight weight)
    {
        if (destinationFp16 == 0)
            throw new ArgumentException("Destination FP16 pointer cannot be null.", nameof(destinationFp16));

        if (weight.QuantizedPointer == 0)
            throw new InvalidOperationException("Cannot dequantize a weight without a quantized source pointer.");

        int totalElements = checked(weight.OutputDim * weight.InputDim);
        long sourceByteLength = checked(Dequantize.RowByteSize(weight.InputDim, weight.QuantizedFormat) * weight.OutputDim);

        if (sourceByteLength > int.MaxValue)
            throw new NotSupportedException(
                $"Metal on-the-fly dequantization source is too large for Span<byte>: {sourceByteLength} bytes.");

        var source = new ReadOnlySpan<byte>((void*)weight.QuantizedPointer, (int)sourceByteLength);
        var destination = new Span<Half>((void*)destinationFp16, totalElements);

        switch (weight.QuantizedFormat)
        {
            case QuantizationType.Q8_0:
                EnsureDivisible(totalElements, 32, weight.QuantizedFormat);
                Dequant.Q8_0ToF16(_context, source, destination, totalElements / 32);
                break;

            case QuantizationType.Q4_0:
                EnsureDivisible(totalElements, 32, weight.QuantizedFormat);
                Dequant.Q4_0ToF16(_context, source, destination, totalElements / 32);
                break;

            case QuantizationType.Q5_0:
                EnsureDivisible(totalElements, 32, weight.QuantizedFormat);
                Dequant.Q5_0ToF16(_context, source, destination, totalElements / 32);
                break;

            case QuantizationType.Q4_K:
                EnsureDivisible(totalElements, 256, weight.QuantizedFormat);
                Dequant.Q4_KToF16(_context, source, destination, totalElements / 256);
                break;

            case QuantizationType.Q5_K:
                EnsureDivisible(totalElements, 256, weight.QuantizedFormat);
                Dequant.Q5_KToF16(_context, source, destination, totalElements / 256);
                break;

            case QuantizationType.Q6_K:
                EnsureDivisible(totalElements, 256, weight.QuantizedFormat);
                Dequant.Q6_KToF16(_context, source, destination, totalElements / 256);
                break;

            case QuantizationType.F16:
                NativeMemory.Copy(
                    source: (void*)weight.QuantizedPointer,
                    destination: (void*)destinationFp16,
                    byteCount: (nuint)(totalElements * sizeof(ushort)));
                break;

            default:
                throw new NotSupportedException(
                    $"On-the-fly Metal dequantization from {weight.QuantizedFormat} to FP16 is not supported.");
        }
    }

    private static void EnsureDivisible(int value, int divisor, QuantizationType format)
    {
        if (value % divisor != 0)
        {
            throw new InvalidOperationException(
                $"Cannot dequantize {format}: element count {value} must be divisible by {divisor}.");
        }
    }

    /// <summary>
    /// Releases all resources used by the <c>MetalTransformerModel</c> instance.
    /// </summary>
    public void Dispose()
    {
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }

    private void Dispose(bool disposing)
    {
        if (_disposed)
            return;

        if (disposing)
        {
            _state.Dispose();
            _weights.Dispose();
            _context.Dispose();
        }

        _disposed = true;
    }

}
