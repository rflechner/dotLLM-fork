using System.Runtime.CompilerServices;
using DotLLM.Metal.Interop;

namespace DotLLM.Metal;

/// <summary>
/// Quantized General Matrix-Vector multiplication accelerated via Metal GPU.
/// Direct translation of quantized_gemv_f32in.cu.
/// </summary>
public static class QuantizedGemv
{
    private const int s_q80BlockSize  = 32;
    private const int s_q80BlockBytes = 34;

    /// <summary>
    /// Multiplies a Q8_0 quantized weight matrix by a float32 input vector.
    /// <c>y[i] = dot(dequant(weight[i, :]), x)</c> for each row <c>i</c>.
    /// </summary>
    /// <param name="ctx">Metal context.</param>
    /// <param name="weight">
    /// Q8_0 weight matrix, row-major.
    /// Length must equal <c>n × (k/32) × 34</c> bytes.
    /// </param>
    /// <param name="x">Float32 input vector, length <c>k</c>.</param>
    /// <param name="y">Float32 output vector, length <c>n</c>.</param>
    /// <param name="n">Number of output rows.</param>
    /// <param name="k">Number of input columns. Must be a multiple of 32.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Q8_0F32In(
        MetalContext       ctx,
        ReadOnlySpan<byte> weight,
        ReadOnlySpan<float> x,
        Span<float>         y,
        int n,
        int k)
    {
        if (k % s_q80BlockSize != 0)
            throw new ArgumentException(
                $"k ({k}) must be a multiple of {s_q80BlockSize} for Q8_0.", nameof(k));

        int blocksPerRow    = k / s_q80BlockSize;
        int expectedWeightLen = n * blocksPerRow * s_q80BlockBytes;
        if (weight.Length != expectedWeightLen)
            throw new ArgumentException(
                $"weight.Length ({weight.Length}) must equal n × (k/32) × {s_q80BlockBytes} = {expectedWeightLen}.",
                nameof(weight));
        if (x.Length != k)
            throw new ArgumentException(
                $"x.Length ({x.Length}) must equal k ({k}).", nameof(x));
        if (y.Length != n)
            throw new ArgumentException(
                $"y.Length ({y.Length}) must equal n ({n}).", nameof(y));

        unsafe
        {
            fixed (byte*  pW = weight)
            fixed (float* pX = x)
            fixed (float* pY = y)
            {
                int code = MetalNative.QuantizedGemvQ8_0F32In(ctx.Handle, pW, pX, pY, n, k);
                if (code != 0)
                    throw new InvalidOperationException(
                        $"Metal quantized_gemv_q8_0_f32in failed with code {code}.");
            }
        }
    }
}
