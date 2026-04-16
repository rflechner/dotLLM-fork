using System.Runtime.CompilerServices;
using DotLLM.Metal.Interop;

namespace DotLLM.Metal;

/// <summary>
/// Element-wise vector multiplication accelerated via Metal GPU.
/// </summary>
public static class Multiply
{
    /// <summary>
    /// Multiplies two vectors element-wise: <c>result[i] = a[i] * b[i]</c>.
    /// </summary>
    /// <param name="ctx">The Metal context that owns the compiled pipeline.</param>
    /// <param name="a">First input vector.</param>
    /// <param name="b">Second input vector.</param>
    /// <param name="result">Output span. Must be at least as long as <paramref name="a"/>.</param>
    /// <exception cref="ArgumentException">
    /// Thrown when input lengths differ or <paramref name="result"/> is too short.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the native Metal kernel returns a non-zero error code.
    /// </exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Execute(MetalContext ctx, ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> result)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Input spans must have the same length.");

        if (result.Length < a.Length)
            throw new ArgumentException("Result span is too small.");

        if (a.Length == 0)
            return;

        unsafe
        {
            fixed (float* pA = a)
            fixed (float* pB = b)
            fixed (float* pR = result)
            {
                int code = MetalNative.MultiplyF32(ctx.Handle, pA, pB, pR, (uint)a.Length);
                if (code != 0)
                    throw new InvalidOperationException($"Metal multiply_f32 failed with code {code}.");
            }
        }
    }
}
