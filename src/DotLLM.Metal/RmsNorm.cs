using System.Runtime.CompilerServices;
using DotLLM.Metal.Interop;

namespace DotLLM.Metal;

/// <summary>
/// RMS Normalization accelerated via Metal GPU.
/// Direct translation of rmsnorm_f32.cu — one threadgroup per token,
/// threads collaborate via simd_sum() and threadgroup memory.
/// </summary>
public static class RmsNorm
{
    /// <summary>
    /// Applies RMS normalization row-wise:
    /// <c>output[t, i] = input[t, i] / rms(input[t]) * weight[i]</c>.
    /// </summary>
    /// <param name="ctx">The Metal context that owns the compiled pipeline.</param>
    /// <param name="input">Input tensor of shape <c>[seqLen × n]</c> (read-only).</param>
    /// <param name="weight">Scale vector of length <c>n</c> (γ parameter).</param>
    /// <param name="output">Output tensor of shape <c>[seqLen × n]</c>, written by the kernel.</param>
    /// <param name="n">Hidden dimension (inner size, e.g. 4096 for Llama-7B).</param>
    /// <param name="seqLen">Number of tokens (outer size).</param>
    /// <param name="eps">Epsilon for numerical stability (e.g. <c>1e-5f</c>).</param>
    /// <exception cref="ArgumentException">
    /// Thrown when buffer lengths are inconsistent with <paramref name="n"/> and <paramref name="seqLen"/>.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the native Metal kernel returns a non-zero error code.
    /// </exception>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Execute(
        MetalContext        ctx,
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> weight,
        Span<float>         output,
        int                 n,
        int                 seqLen,
        float               eps = 1e-5f)
    {
        if (weight.Length != n)
            throw new ArgumentException($"weight.Length ({weight.Length}) must equal n ({n}).", nameof(weight));
        if (input.Length != seqLen * n)
            throw new ArgumentException($"input.Length ({input.Length}) must equal seqLen × n ({seqLen * n}).", nameof(input));
        if (output.Length != seqLen * n)
            throw new ArgumentException($"output.Length ({output.Length}) must equal seqLen × n ({seqLen * n}).", nameof(output));

        if (seqLen == 0) return;

        unsafe
        {
            fixed (float* pIn  = input)
            fixed (float* pW   = weight)
            fixed (float* pOut = output)
            {
                int code = MetalNative.RmsNormF32(ctx.Handle, pIn, pW, pOut, n, seqLen, eps);
                if (code != 0)
                    throw new InvalidOperationException($"Metal rmsnorm_f32 failed with code {code}.");
            }
        }
    }
}
