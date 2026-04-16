using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace DotLLM.Metal;

/// <summary>
/// Multiplies 2 vectors in parallel on the GPU on Metal.
/// </summary>
public static partial class Multiply
{
    /// <summary>
    /// Multiplies two vectors element-wise in parallel using Metal GPU acceleration.
    /// The result is stored in the provided output span.
    /// </summary>
    /// <param name="a">The first input vector as a read-only span of floats.</param>
    /// <param name="b">The second input vector as a read-only span of floats.</param>
    /// <param name="result">
    /// A span of floats where the result of the addition will be stored.
    /// Must have a length that is equal to or greater than the length of the input vectors.
    /// </param>
    /// <exception cref="ArgumentException">
    /// Thrown when the input spans <paramref name="a"/> and <paramref name="b"/> do not have the same length,
    /// or when <paramref name="result"/> does not have sufficient length to store the output.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the underlying Metal GPU operation fails.
    /// </exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Execute(ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException("Input spans must have the same length.");
        }

        if (result.Length < a.Length)
        {
            throw new ArgumentException("Result span is too small.");
        }

        if (a.Length == 0)
        {
            return;
        }

        unsafe
        {
            ref float left = ref MemoryMarshal.GetReference(a);
            ref float right = ref MemoryMarshal.GetReference(b);
            ref float resultRef = ref MemoryMarshal.GetReference(result);
            fixed (float* l = &left)
            fixed (float* r = &right)
            fixed (float* output = &resultRef)
            {
                int code = NativeMethods.dotllm_metal_multiply_f32(l, r, output, (uint)a.Length);
                if (code != 0)
                {
                    throw new InvalidOperationException($"Metal add failed with code {code}.");
                }
            }
        }
    }

    private static partial class NativeMethods
    {
        [LibraryImport("dotllmmetal", EntryPoint = "dotllm_metal_multiply_f32")]
        internal static unsafe partial int dotllm_metal_multiply_f32(
            float* a,
            float* b,
            float* result,
            uint length);
    }
}
