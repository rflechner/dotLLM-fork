using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;
using DotLLM.Metal.Weights.Strategies;
using DotLLM.Models.Gguf;

namespace DotLLM.Metal.Weights;

/// <summary>
/// All model parameters resolved against a memory-mapped GGUF file, ready for
/// the Metal forward pass. Owns the underlying <see cref="GgufFile"/> (so the
/// mmap stays alive as long as any pointer here is in use) and any FP16 buffers
/// the loading strategy allocated.
/// </summary>
public sealed class MetalWeights : IDisposable
{
    private readonly GgufFile _gguf;
    private readonly nint[]   _ownedFp16Buffers;   // freed in Dispose
    private bool _disposed;

    /// <summary>Per-layer weights (length = <see cref="ModelConfig.NumLayers"/>).</summary>
    public MetalLayerWeights[] Layers           { get; }

    /// <summary>Token embedding table.</summary>
    public LoadedWeight        TokenEmbedding   { get; }

    /// <summary>LM head (output projection). Falls back to <see cref="TokenEmbedding"/> when tied.</summary>
    public LoadedWeight        LmHead           { get; }

    /// <summary>Final RMSNorm scale before the LM head (FP16, mmap-direct).</summary>
    public nint                OutputNormWeight { get; }

    /// <summary>Architecture configuration extracted from the GGUF metadata.</summary>
    public ModelConfig         Config           { get; }

    /// <summary>Name of the <see cref="IWeightLoadStrategy"/> used at load time (for diagnostics).</summary>
    public string              StrategyName     { get; }

    private MetalWeights(
        GgufFile gguf,
        ModelConfig config,
        MetalLayerWeights[] layers,
        LoadedWeight tokenEmbedding,
        LoadedWeight lmHead,
        nint outputNormWeight,
        nint[] ownedFp16Buffers,
        string strategyName)
    {
        _gguf             = gguf;
        Config            = config;
        Layers            = layers;
        TokenEmbedding    = tokenEmbedding;
        LmHead            = lmHead;
        OutputNormWeight  = outputNormWeight;
        _ownedFp16Buffers = ownedFp16Buffers;
        StrategyName      = strategyName;
    }

    /// <summary>
    /// Loads model weights from a GGUF file using the specified <paramref name="strategy"/>
    /// to decide how each quantized projection is stored.
    /// </summary>
    /// <param name="gguf">GGUF model file.</param>
    /// <param name="ctx">Metal context (for kernel dispatch during dequantization).</param>
    /// <param name="strategy">Per-tensor storage policy.</param>
    public static MetalWeights LoadFromGguf(
        GgufFile gguf, MetalContext ctx, IWeightLoadStrategy strategy)
    {
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        // Track every FP16 buffer the strategy allocates so we can free them later.
        var owned = new List<nint>(capacity: config.NumLayers * 7 + 2);

        try
        {
            var layers = new MetalLayerWeights[config.NumLayers];
            for (int i = 0; i < config.NumLayers; i++)
                layers[i] = LoadLayer(ctx, gguf, i, strategy, owned);

            // Token embedding — present in every model.
            var tokenEmbedding = LoadProjection(ctx, gguf, "token_embd.weight", strategy, owned);

            // Output norm — always FP16, mmap-direct.
            nint outNorm = ResolvePointer(gguf, "output_norm.weight");

            // LM head — falls back to token_embd when tied (Llama 3.2 1B/3B, small Qwen).
            bool hasOutput = gguf.TensorsByName.ContainsKey("output.weight");
            var lmHead = (config.TiedEmbeddings || !hasOutput)
                ? tokenEmbedding
                : LoadProjection(ctx, gguf, "output.weight", strategy, owned);

            return new MetalWeights(
                gguf, config, layers, tokenEmbedding, lmHead, outNorm,
                owned.ToArray(), strategy.Name);
        }
        catch
        {
            // If anything goes wrong mid-load, free what we've allocated and
            // dispose the gguf. Don't leak.
            foreach (var ptr in owned)
                FreeFp16(ptr);
            gguf.Dispose();
            throw;
        }
    }

    /// <summary>Frees any FP16 buffers the strategy allocated and closes the GGUF mmap.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        foreach (var ptr in _ownedFp16Buffers)
            FreeFp16(ptr);

        _gguf.Dispose();
    }

    // ────────────────────────────────────────────────────────────────────────
    // Loading helpers
    // ────────────────────────────────────────────────────────────────────────

    private static MetalLayerWeights LoadLayer(
        MetalContext ctx, GgufFile gguf, int i,
        IWeightLoadStrategy strategy, List<nint> owned)
    {
        return new MetalLayerWeights
        {
            // Attention projections
            Q = LoadProjection(ctx, gguf, $"blk.{i}.attn_q.weight",      strategy, owned),
            K = LoadProjection(ctx, gguf, $"blk.{i}.attn_k.weight",      strategy, owned),
            V = LoadProjection(ctx, gguf, $"blk.{i}.attn_v.weight",      strategy, owned),
            O = LoadProjection(ctx, gguf, $"blk.{i}.attn_output.weight", strategy, owned),

            // FFN projections
            Gate = LoadProjection(ctx, gguf, $"blk.{i}.ffn_gate.weight", strategy, owned),
            Up   = LoadProjection(ctx, gguf, $"blk.{i}.ffn_up.weight",   strategy, owned),
            Down = LoadProjection(ctx, gguf, $"blk.{i}.ffn_down.weight", strategy, owned),

            // RMSNorm scales — required.
            AttnNormWeight = ResolvePointer(gguf, $"blk.{i}.attn_norm.weight"),
            FfnNormWeight  = ResolvePointer(gguf, $"blk.{i}.ffn_norm.weight"),

            // QK-norm — optional (Qwen3, Cohere).
            QNormWeight = TryResolvePointer(gguf, $"blk.{i}.attn_q_norm.weight"),
            KNormWeight = TryResolvePointer(gguf, $"blk.{i}.attn_k_norm.weight"),

            // Biases — optional (Qwen2 Q/K/V).
            QBias    = TryResolvePointer(gguf, $"blk.{i}.attn_q.bias"),
            KBias    = TryResolvePointer(gguf, $"blk.{i}.attn_k.bias"),
            VBias    = TryResolvePointer(gguf, $"blk.{i}.attn_v.bias"),
            OBias    = TryResolvePointer(gguf, $"blk.{i}.attn_output.bias"),
            GateBias = TryResolvePointer(gguf, $"blk.{i}.ffn_gate.bias"),
            UpBias   = TryResolvePointer(gguf, $"blk.{i}.ffn_up.bias"),
            DownBias = TryResolvePointer(gguf, $"blk.{i}.ffn_down.bias"),
        };
    }

    /// <summary>
    /// Resolves a projection tensor and hands it to the strategy.
    /// Tracks any FP16 buffer the strategy allocates so it can be freed later.
    /// </summary>
    private static LoadedWeight LoadProjection(
        MetalContext ctx, GgufFile gguf, string tensorName,
        IWeightLoadStrategy strategy, List<nint> owned)
    {
        var src    = BuildTensorSource(gguf, tensorName);
        var loaded = strategy.Load(ctx, src);

        if (loaded.OwnsFp16Buffer && loaded.Fp16Pointer != 0)
            owned.Add(loaded.Fp16Pointer);

        return loaded;
    }

    /// <summary>Builds the <see cref="TensorSource"/> handed to the strategy.</summary>
    private static TensorSource BuildTensorSource(GgufFile gguf, string tensorName)
    {
        var desc = gguf.TensorsByName[tensorName];

        // GGUF projection tensors are stored as [outputDim, inputDim] (row-major).
        // 1-D tensors (norms, biases) collapse inputDim to 1.
//        int outputDim = (int)desc.Shape.Dimensions[0];
//        int inputDim  = desc.Shape.Dimensions.Length > 1 ? (int)desc.Shape.Dimensions[1] : 1;

        int outputDim;
        int inputDim;

        if (desc.Shape.Dimensions.Length > 1)
        {
            outputDim = desc.Shape.Dimensions[1];
            inputDim = desc.Shape.Dimensions[0];
        }
        else
        {
            outputDim = desc.Shape.Dimensions[0];
            inputDim = 1;
        }

        // Total bytes = bytes-per-row × number-of-rows.
        long byteLen = Dequantize.RowByteSize(inputDim, desc.QuantizationType) * outputDim;

        nint ptr = gguf.DataBasePointer + (nint)desc.DataOffset;

        return new TensorSource(
            mmapPointer: ptr,
            byteLength:  byteLen,
            format:      desc.QuantizationType,
            outputDim:   outputDim,
            inputDim:    inputDim);
    }

    /// <summary>Returns the absolute mmap pointer for a tensor that must exist.</summary>
    private static nint ResolvePointer(GgufFile gguf, string tensorName)
    {
        var desc = gguf.TensorsByName[tensorName];
        return gguf.DataBasePointer + (nint)desc.DataOffset;
    }

    /// <summary>Returns 0 when the tensor is absent (for optional tensors).</summary>
    private static nint TryResolvePointer(GgufFile gguf, string tensorName)
    {
        return gguf.TensorsByName.TryGetValue(tensorName, out var desc)
            ? gguf.DataBasePointer + (nint)desc.DataOffset
            : 0;
    }

    private static unsafe void FreeFp16(nint ptr)
    {
        if (ptr != 0)
            NativeMemory.AlignedFree((void*)ptr);
    }
}
