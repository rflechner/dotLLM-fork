using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Models.Gguf;

namespace DotLLM.Metal.Helpers;

/// <summary>
/// Provides helper methods for loading GGUF tensors into memory for processing.
/// Specifically, it handles conversion of 1-dimensional tensors to FP16 when required.
/// GGUF tensors may be stored in FP32 and require conversion for compatibility with certain processing kernels.
/// </summary>
public static class LoadGgufHelpers
{
    /// <summary>
    /// Loads a 1-D GGUF tensor as FP16.
    /// Norm weights and biases are commonly stored as FP32 in GGUF, while Metal
    /// kernels consume FP16 pointers.
    /// </summary>
    public static unsafe nint LoadVectorFp16(GgufFile gguf, string tensorName, List<nint> owned)
    {
        var desc = gguf.TensorsByName[tensorName];
        int elementCount = checked((int)desc.Shape.ElementCount);
        nint src = gguf.DataBasePointer + (nint)desc.DataOffset;

        switch (desc.QuantizationType)
        {
            case QuantizationType.F16:
                return src;

            case QuantizationType.F32:
            {
                nint dst = AllocFp16Aligned(elementCount);
                var srcSpan = new ReadOnlySpan<float>((void*)src, elementCount);
                var dstSpan = new Span<Half>((void*)dst, elementCount);

                for (int i = 0; i < elementCount; i++)
                    dstSpan[i] = (Half)srcSpan[i];

                owned.Add(dst);
                return dst;
            }

            default:
                throw new NotSupportedException(
                    $"Tensor '{tensorName}' has unsupported vector format {desc.QuantizationType}. " +
                    "Expected F16 or F32.");
        }
    }

    /// <summary>
    /// Loads an optional 1-D GGUF tensor as FP16. Returns 0 when absent.
    /// </summary>
    public static nint TryLoadVectorFp16(GgufFile gguf, string tensorName, List<nint> owned)
    {
        return gguf.TensorsByName.ContainsKey(tensorName)
            ? LoadVectorFp16(gguf, tensorName, owned)
            : 0;
    }

    /// <summary>
    /// Allocates memory aligned to a 64-byte boundary, sufficient to store the given number of FP16 elements.
    /// </summary>
    /// <param name="elementCount">The number of FP16 elements to allocate memory for.</param>
    /// <returns>A pointer to the allocated, aligned memory block.</returns>
    public static unsafe nint AllocFp16Aligned(int elementCount)
    {
        nuint bytes = checked((nuint)elementCount * sizeof(ushort));
        return (nint)NativeMemory.AlignedAlloc(bytes, alignment: 64);
    }

    /// <summary>
    /// Frees a memory block allocated for a 1-D FP16 tensor, ensuring proper deallocation of aligned memory.
    /// </summary>
    /// <param name="ptr">The pointer to the memory block to be freed. If the pointer is zero, the method does nothing.</param>
    public static unsafe void FreeFp16(nint ptr)
    {
        if (ptr != 0)
            NativeMemory.AlignedFree((void*)ptr);
    }
}
