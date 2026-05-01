using DotLLM.Core.Tensors;

namespace DotLLM.Metal;

/// <summary>
/// Tensor backed by Metal-accessible native memory.
/// By default this is a non-owning view over memory managed elsewhere,
/// for example MetalForwardState.
/// </summary>
public sealed class MetalTensor : ITensor
{
    private nint _ptr;

    /// <inheritdoc/>
    public TensorShape Shape { get; }

    /// <inheritdoc/>
    public DType DType { get; }

    /// <inheritdoc/>
    public int DeviceId { get; }

    /// <inheritdoc/>
    public nint DataPointer => _ptr;

    /// <inheritdoc/>
    public TensorMetadata Metadata => new(Shape, DType, DeviceId, _ptr);

    /// <inheritdoc/>
    public long ElementCount => Shape.ElementCount;

    /// <inheritdoc/>
    public long ByteCount { get; }

    private readonly bool _ownsMemory;

    /// <inheritdoc/>
    public MetalTensor(nint ptr, int elementCount, int deviceId = 0)
        : this(new TensorShape(elementCount), DType.Float32, deviceId, ptr, ownsMemory: false)
    {
    }

    /// <summary>
    /// Create a MetalTensor from a native pointer.
    /// </summary>
    /// <param name="shape"></param>
    /// <param name="dtype"></param>
    /// <param name="deviceId"></param>
    /// <param name="ptr"></param>
    /// <param name="ownsMemory"></param>
    /// <exception cref="ArgumentException"></exception>
    public MetalTensor(TensorShape shape, DType dtype, int deviceId, nint ptr, bool ownsMemory = false)
    {
        if (ptr == 0)
            throw new ArgumentException("Tensor pointer cannot be null.", nameof(ptr));

        Shape = shape;
        DType = dtype;
        DeviceId = deviceId;
        _ptr = ptr;
        _ownsMemory = ownsMemory;
        ByteCount = dtype.ComputeByteCount(shape.ElementCount);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (!_ownsMemory)
        {
            _ptr = 0;
            return;
        }

        nint ptr = Interlocked.Exchange(ref _ptr, 0);
        if (ptr == 0)
            return;

        // TODO: free Metal-owned memory here if/when MetalTensor supports ownership.
        // For now, avoid freeing because forward-state buffers are owned elsewhere.
    }
}
