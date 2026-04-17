using System.Runtime.CompilerServices;
using DotLLM.Metal.Interop;

namespace DotLLM.Metal;

/// <summary>
/// In-place bias addition accelerated via Metal GPU.
/// </summary>
public static class BiasAdd
{
    /// <summary>
    /// Adds a bias vector to each token in-place: <c>output[t, i] += bias[i]</c>.
    /// </summary>
    /// <param name="ctx">The Metal context that owns the compiled pipeline.</param>
    /// <param name="output">
    /// Flat buffer of shape <c>[seqLen × dim]</c>, modified in-place.
    /// Must have length equal to <paramref name="dim"/> × <paramref name="seqLen"/>.
    /// </param>
    /// <param name="bias">Bias vector of length <paramref name="dim"/>.</param>
    /// <param name="dim">Feature dimension (inner size).</param>
    /// <param name="seqLen">Number of tokens (outer size).</param>
    /// <exception cref="ArgumentException">
    /// Thrown when buffer lengths are inconsistent with <paramref name="dim"/> and <paramref name="seqLen"/>.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the native Metal kernel returns a non-zero error code.
    /// </exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Execute(MetalContext ctx, Span<float> output, ReadOnlySpan<float> bias, int dim, int seqLen)
    {
        if (bias.Length != dim)
            throw new ArgumentException($"Bias length {bias.Length} does not match dim {dim}.");

        if (output.Length != dim * seqLen)
            throw new ArgumentException($"Output length {output.Length} must equal dim × seqLen ({dim * seqLen}).");

        if (output.IsEmpty)
            return;

        unsafe
        {
            fixed (float* pOut  = output)
            fixed (float* pBias = bias)
            {
                int code = MetalNative.BiasAddF32(ctx.Handle, pOut, pBias, (uint)dim, (uint)seqLen);
                if (code != 0)
                    throw new InvalidOperationException($"Metal bias_add_f32 failed with code {code}.");
            }
        }
    }
}
