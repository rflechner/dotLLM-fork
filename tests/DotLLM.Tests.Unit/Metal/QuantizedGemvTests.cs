using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Metal;
using Xunit;

namespace DotLLM.Tests.Unit.Metal;

/// <summary>
/// Tests for <see cref="QuantizedGemv.Q8_0F32In"/>.
/// CPU reference: dequantize weight matrix with <see cref="Dequantize.ToFloat32"/>,
/// then compute the dot product in float32.
/// </summary>
public sealed class QuantizedGemvTests
{
    // ── CPU reference ─────────────────────────────────────────────────────────

    /// <summary>
    /// Dequantizes the Q8_0 weight matrix to float32, then computes y = W * x.
    /// </summary>
    private static float[] CpuGemv(byte[] weight, float[] x, int n, int k)
    {
        // Step 1: dequantize W → float32
        float[] wFloat = new float[n * k];
        unsafe
        {
            fixed (byte* p = weight)
                Dequantize.ToFloat32((nint)p, n * k, QuantizationType.Q8_0, wFloat);
        }

        // Step 2: y[i] = dot(W[i,:], x)
        float[] y = new float[n];
        for (int i = 0; i < n; i++)
        {
            float acc = 0f;
            for (int j = 0; j < k; j++)
                acc += wFloat[i * k + j] * x[j];
            y[i] = acc;
        }
        return y;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    // Build a Q8_0 weight matrix: n rows × (k/32) blocks × 34 bytes.
    private static byte[] BuildWeight(int n, int k, int seed)
    {
        const int BlockSize = 32, BlockBytes = 34;
        int blocksPerRow = k / BlockSize;
        byte[] buf = new byte[n * blocksPerRow * BlockBytes];
        var rng = new Random(seed);

        for (int row = 0; row < n; row++)
        for (int b = 0; b < blocksPerRow; b++)
        {
            int off = (row * blocksPerRow + b) * BlockBytes;
            MemoryMarshal.Write(buf.AsSpan(off, 2), (Half)(rng.NextSingle() * 0.1f + 0.01f));
            for (int j = 0; j < BlockSize; j++)
                buf[off + 2 + j] = (byte)(sbyte)rng.Next(-127, 128);
        }
        return buf;
    }

    private static float[] BuildX(int k, int seed)
    {
        var rng = new Random(seed);
        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;
        return x;
    }

    // The GPU accumulates in float32 with a different summation order than the
    // CPU reference, so we allow a small relative tolerance.
    private const float Tol = 1e-3f;

    private static void AssertEqual(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], Tol);
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    [Fact]
    public void Q8_0_SingleRow_MatchesCpu()
    {
        const int n = 1, k = 32;
        byte[]  weight   = BuildWeight(n, k, seed: 1);
        float[] x        = BuildX(k, seed: 2);
        float[] expected = CpuGemv(weight, x, n, k);
        float[] y        = new float[n];

        using var ctx = new MetalContext();
        QuantizedGemv.Q8_0F32In(ctx, weight, x, y, n, k);

        AssertEqual(expected, y);
    }

    [Fact]
    public void Q8_0_MultipleRows_MatchesCpu()
    {
        const int n = 8, k = 128;
        byte[]  weight   = BuildWeight(n, k, seed: 3);
        float[] x        = BuildX(k, seed: 4);
        float[] expected = CpuGemv(weight, x, n, k);
        float[] y        = new float[n];

        using var ctx = new MetalContext();
        QuantizedGemv.Q8_0F32In(ctx, weight, x, y, n, k);

        AssertEqual(expected, y);
    }

    [Fact]
    public void Q8_0_LargerMatrix_MatchesCpu()
    {
        // Realistic hidden-dim size to stress the grid-stride loop.
        const int n = 32, k = 512;
        byte[]  weight   = BuildWeight(n, k, seed: 5);
        float[] x        = BuildX(k, seed: 6);
        float[] expected = CpuGemv(weight, x, n, k);
        float[] y        = new float[n];

        using var ctx = new MetalContext();
        QuantizedGemv.Q8_0F32In(ctx, weight, x, y, n, k);

        AssertEqual(expected, y);
    }

    [Fact]
    public void Q8_0_ZeroWeights_OutputIsZero()
    {
        const int n = 4, k = 64;
        int blocksPerRow = k / 32;
        byte[] weight = new byte[n * blocksPerRow * 34];
        // Write scale = 1.0 at each block header, leave int8 weights as 0.
        for (int row = 0; row < n; row++)
        for (int b = 0; b < blocksPerRow; b++)
        {
            int off = (row * blocksPerRow + b) * 34;
            MemoryMarshal.Write(weight.AsSpan(off, 2), (Half)1.0f);
        }

        float[] x = BuildX(k, seed: 7);
        float[] y = new float[n];

        using var ctx = new MetalContext();
        QuantizedGemv.Q8_0F32In(ctx, weight, x, y, n, k);

        foreach (float v in y) Assert.Equal(0f, v, 1e-6f);
    }

    [Fact]
    public void Q8_0_ZeroInput_OutputIsZero()
    {
        const int n = 4, k = 64;
        byte[]  weight = BuildWeight(n, k, seed: 8);
        float[] x      = new float[k]; // all zeros
        float[] y      = new float[n];

        using var ctx = new MetalContext();
        QuantizedGemv.Q8_0F32In(ctx, weight, x, y, n, k);

        foreach (float v in y) Assert.Equal(0f, v, 1e-6f);
    }
}
