using System.Runtime.CompilerServices;
using DotLLM.Metal.Interop;

namespace DotLLM.Metal;

/// <summary>
/// Softmax activation function.
/// </summary>
public static class Softmax
{
    /// <summary>
    /// Executes the softmax operation on the input values, computing the probabilities
    /// for each element such that they are normalized and sum to 1.
    /// </summary>
    /// <param name="ctx">The Metal context that owns the compiled pipeline.</param>
    /// <param name="input">
    /// A read-only span of input values representing scores or logits for which the softmax probabilities will be computed.
    /// </param>
    /// <param name="result">
    /// A writable span to store the resulting softmax probabilities. Must have the same length as the input.
    /// </param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Execute(MetalContext ctx, ReadOnlySpan<float> input, Span<float> result)
    {
        if (result.Length < input.Length)
            throw new ArgumentException("Result span is too small.");

        if (input.Length == 0)
            return;

        unsafe
        {
            fixed (float* pA = input)
            fixed (float* pR = result)
            {
                int code = MetalNative.SoftmaxF32(ctx.Handle, pA, pR, (uint)input.Length);
                if (code != 0)
                    throw new InvalidOperationException($"Metal multiply_f32 failed with code {code}.");
            }
        }
    }
}

