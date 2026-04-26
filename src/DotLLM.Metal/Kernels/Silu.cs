using System.Runtime.CompilerServices;
using DotLLM.Metal.Interop;

namespace DotLLM.Metal.Kernels;

/// <summary>
/// Provides SILU Kernel with Metal GPU acceleration.
/// </summary>
public static class Silu
{
    /// <summary>
    /// Adds two vectors element-wise: <c>result[i] = a[i] + b[i]</c>.
    /// </summary>
    /// <param name="ctx">The Metal context that owns the compiled pipeline.</param>
    /// <param name="input">First input vector.</param>
    /// <param name="result">Output span. Must be at least as long as <paramref name="input"/>.</param>
    /// <exception cref="ArgumentException">
    /// Thrown when input lengths differ or <paramref name="result"/> is too short.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the native Metal kernel returns a non-zero error code.
    /// </exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Execute(MetalContext ctx, ReadOnlySpan<float> input, Span<float> result)
    {
        if (result.Length < input.Length)
            throw new ArgumentException("Result span is too small.");

        if (input.Length == 0)
            return;

        unsafe
        {
            fixed (float* pInput = input)
            fixed (float* pR = result)
            {
                int code = MetalNative.SiluF32(ctx.Handle, pInput, pR, (uint)input.Length);
                if (code != 0)
                    throw new InvalidOperationException($"Metal silu_f32 failed with code {code}.");
            }
        }
    }
}
