using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Metal.Interop;

namespace DotLLM.Metal;

/// <summary>
/// KV-cache backed by persistent <c>MTLResourceStorageModeShared</c> MTLBuffers.
/// On Apple Silicon, Shared storage is the same physical memory for CPU and GPU —
/// the C# forward pass writes K/V
/// and the attention kernel reads the same bytes without any copy or synchronisation.
/// </summary>
public sealed unsafe class MetalKvCache : IKvCache
{
    private nint _handle;   // dotllm_metal_kvcache*
    private readonly int _numLayers;
    private readonly int _kvStride;   // numKvHeads * headDim
    private readonly int _maxSeqLen;
    private int _currentLength;
    private bool _disposed;

    /// <summary>Opaque native handle used by the attention kernels.</summary>
    internal nint Handle => _handle;

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;

    /// <inheritdoc/>
    public int MaxLength => _maxSeqLen;

    /// <summary>Bytes allocated on the GPU (two FP16 buffers per layer).</summary>
    public long AllocatedBytes => (long)_numLayers * 2 * _maxSeqLen * _kvStride * sizeof(ushort);

    /// <summary>
    /// Allocates persistent MTLBuffers for every layer.
    /// The Metal context must outlive this cache.
    /// </summary>
    public MetalKvCache(MetalContext ctx, int numLayers, int numKvHeads, int headDim, int maxSeqLen)
    {
        _numLayers = numLayers;
        _kvStride  = numKvHeads * headDim;
        _maxSeqLen = maxSeqLen;

        _handle = MetalNative.KvCacheCreate(ctx.Handle, numLayers, numKvHeads, headDim, maxSeqLen);
        if (_handle == 0)
            throw new InvalidOperationException(
                "Failed to create Metal KV-cache. Possible OOM or invalid geometry.");
    }

    /// <summary>Finalizer — releases native handle if <see cref="Dispose()"/> was not called.</summary>
    ~MetalKvCache() => Dispose(false);

    /// <summary>
    /// Copies K and V projections for the current tokens into the cache at the given positions.
    /// Called once per layer per forward step. Zero GPU involvement — pure CPU memcpy into
    /// the MTLBuffer's shared memory region.
    /// </summary>
    /// <param name="layer">Transformer layer index.</param>
    /// <param name="kSrc">Pointer to K projection, layout <c>[seqLen, kvStride]</c> FP16.</param>
    /// <param name="vSrc">Pointer to V projection, layout <c>[seqLen, kvStride]</c> FP16.</param>
    /// <param name="positions">Position indices for the current tokens (sorted ascending).</param>
    private void WriteKV(
        int layer,
        nint kSrc,
        nint vSrc,
        ReadOnlySpan<int> positions)
    {
        if (positions.IsEmpty)
            return;

        nint kBasePtr = (nint)MetalNative.KvCacheKeyPtr(_handle, layer);
        nint vBasePtr = (nint)MetalNative.KvCacheValuePtr(_handle, layer);

        int elementsPerToken = _kvStride;
        int sourceElementCount = positions.Length * elementsPerToken;

        var kSource = new ReadOnlySpan<ushort>((void*)kSrc, sourceElementCount);
        var vSource = new ReadOnlySpan<ushort>((void*)vSrc, sourceElementCount);

        for (int i = 0; i < positions.Length; i++)
        {
            int position = positions[i];

            WriteKV(
                kBasePtr,
                vBasePtr,
                position,
                elementsPerToken,
                kSource.Slice(i * elementsPerToken, elementsPerToken),
                vSource.Slice(i * elementsPerToken, elementsPerToken));
        }

        int newLen = positions[^1] + 1;
        if (newLen > _currentLength)
        {
            _currentLength = newLen;
            MetalNative.KvCacheSetCurrentLength(_handle, _currentLength);
        }
    }

    private static void WriteKV(
        nint kBasePtr,
        nint vBasePtr,
        int currentTokenIndex,
        int elementsPerToken,
        ReadOnlySpan<ushort> kData,
        ReadOnlySpan<ushort> vData)
    {
        if (kData.Length != elementsPerToken || vData.Length != elementsPerToken)
            throw new ArgumentException("Invalid KV size");

        var kSpan = new Span<ushort>((void*)kBasePtr, elementsPerToken * (currentTokenIndex + 1));
        var vSpan = new Span<ushort>((void*)vBasePtr, elementsPerToken * (currentTokenIndex + 1));

        int offset = currentTokenIndex * elementsPerToken;

        kData.CopyTo(kSpan.Slice(offset, elementsPerToken));
        vData.CopyTo(vSpan.Slice(offset, elementsPerToken));
    }

    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
    {
        WriteKV(layerIndex, keys.DataPointer, values.DataPointer, positions);
    }

    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
    {
        var kRef = new TensorRef(positions.Length, _kvStride, DType.Float16, -1, keys.DataPointer);
        var vRef = new TensorRef(positions.Length, _kvStride, DType.Float16, -1, values.DataPointer);
        Update(kRef, vRef, positions, layerIndex);
    }

    /// <inheritdoc/>
    public TensorRef GetKeysRef(int layerIndex)
    {
        nint ptr = (nint)MetalNative.KvCacheKeyPtr(_handle, layerIndex);
        return new TensorRef(_currentLength, _kvStride, DType.Float16, -1, ptr);
    }

    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex)
    {
        nint ptr = (nint)MetalNative.KvCacheValuePtr(_handle, layerIndex);
        return new TensorRef(_currentLength, _kvStride, DType.Float16, -1, ptr);
    }

    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex)
    {
        nint ptr = (nint)MetalNative.KvCacheKeyPtr(_handle, layerIndex);
        return new MetalTensor(new TensorShape(_currentLength, _kvStride), DType.Float16, -1, ptr, ownsMemory: false);
    }

    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex)
    {
        nint ptr = (nint)MetalNative.KvCacheValuePtr(_handle, layerIndex);
        return new MetalTensor(new TensorShape(_currentLength, _kvStride), DType.Float16, -1, ptr, ownsMemory: false);
    }

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        if ((uint)length > (uint)_currentLength)
            throw new ArgumentOutOfRangeException(nameof(length));
        _currentLength = length;
        MetalNative.KvCacheSetCurrentLength(_handle, _currentLength);
    }

    /// <summary>
    /// Resets the visible length to an arbitrary value (used by prefix-cache reuse).
    /// </summary>
    internal void SetCurrentLength(int length)
    {
        if ((uint)length > (uint)_maxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(length));
        _currentLength = length;
        MetalNative.KvCacheSetCurrentLength(_handle, _currentLength);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    private void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (_handle != 0)
        {
            MetalNative.KvCacheDestroy(_handle);
            _handle = 0;
        }
    }
}
