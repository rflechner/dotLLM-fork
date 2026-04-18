using DotLLM.Metal;
using Xunit;

namespace DotLLM.Tests.Unit.Metal;

public sealed class RmsNormTests
{
    // CPU reference for a full [seqLen × n] buffer.
    // The CPU kernel operates row by row, so we call it once per token.
    private static float[] CpuExpected(float[] input, float[] weight, int n, int seqLen, float eps)
    {
        float[] expected = new float[seqLen * n];
        for (int t = 0; t < seqLen; t++)
        {
            var row    = input.AsSpan(t * n, n);
            var rowOut = expected.AsSpan(t * n, n);
            DotLLM.Cpu.Kernels.RmsNorm.Execute(row, weight, eps, rowOut);
        }
        return expected;
    }

    // ── Single token ─────────────────────────────────────────────────────────

    [Fact]
    public void SingleToken_MatchesCpu()
    {
        const int n = 8;
        const float eps = 1e-5f;
        var rng = new Random(1);

        float[] input  = new float[n];
        float[] weight = new float[n];
        for (int i = 0; i < n; i++) { input[i] = rng.NextSingle() * 2f - 1f; weight[i] = rng.NextSingle() + 0.5f; }

        float[] expected = CpuExpected(input, weight, n, 1, eps);
        float[] output   = new float[n];

        using var ctx = new MetalContext();
        RmsNorm.Execute(ctx, input, weight, output, n, 1, eps);

        for (int i = 0; i < n; i++) Assert.Equal(expected[i], output[i], 1e-4f);
    }

    // ── Multiple tokens ───────────────────────────────────────────────────────

    [Fact]
    public void MultipleTokens_MatchesCpu()
    {
        const int n = 16, seqLen = 4;
        const float eps = 1e-5f;
        var rng = new Random(2);

        float[] input  = new float[seqLen * n];
        float[] weight = new float[n];
        for (int i = 0; i < input.Length; i++) input[i]  = rng.NextSingle() * 2f - 1f;
        for (int i = 0; i < n; i++)            weight[i] = rng.NextSingle() + 0.5f;

        float[] expected = CpuExpected(input, weight, n, seqLen, eps);
        float[] output   = new float[seqLen * n];

        using var ctx = new MetalContext();
        RmsNorm.Execute(ctx, input, weight, output, n, seqLen, eps);

        for (int i = 0; i < output.Length; i++) Assert.Equal(expected[i], output[i], 1e-4f);
    }

    // ── Uniform input: analytically known result ──────────────────────────────
    // x = [1, 1, …, 1], weight = [1, 1, …, 1]
    // mean_sq = 1, rms = sqrt(1 + eps), output[i] = 1 / sqrt(1 + eps)

    [Fact]
    public void UniformInput_OutputIsApproximatelyOne()
    {
        const int n = 8;
        const float eps = 1e-5f;

        float[] input  = Enumerable.Repeat(1f, n).ToArray();
        float[] weight = Enumerable.Repeat(1f, n).ToArray();
        float[] output = new float[n];

        using var ctx = new MetalContext();
        RmsNorm.Execute(ctx, input, weight, output, n, 1, eps);

        float expectedVal = 1f / MathF.Sqrt(1f + eps); // exact analytical value
        for (int i = 0; i < n; i++) Assert.Equal(expectedVal, output[i], 1e-5f);
    }

    // ── Weight scaling applied correctly ──────────────────────────────────────
    // If weight[i] = 2 for all i, output[i] must be ×2 compared to weight = 1.

    [Fact]
    public void WeightScaling_DoublesOutput()
    {
        const int n = 8;
        const float eps = 1e-5f;
        var rng = new Random(3);

        float[] input   = new float[n];
        float[] weight1 = Enumerable.Repeat(1f, n).ToArray();
        float[] weight2 = Enumerable.Repeat(2f, n).ToArray();
        for (int i = 0; i < n; i++) input[i] = rng.NextSingle() * 2f - 1f;

        float[] out1 = new float[n];
        float[] out2 = new float[n];

        using var ctx = new MetalContext();
        RmsNorm.Execute(ctx, input, weight1, out1, n, 1, eps);
        RmsNorm.Execute(ctx, input, weight2, out2, n, 1, eps);

        for (int i = 0; i < n; i++) Assert.Equal(out1[i] * 2f, out2[i], 1e-4f);
    }

    // ── Epsilon prevents division by zero ─────────────────────────────────────

    [Fact]
    public void NearZeroInput_EpsilonPreventsNaN()
    {
        const int n = 8;
        const float eps = 1e-5f;

        float[] input  = new float[n]; // all zeros
        float[] weight = Enumerable.Repeat(1f, n).ToArray();
        float[] output = new float[n];

        using var ctx = new MetalContext();
        RmsNorm.Execute(ctx, input, weight, output, n, 1, eps);

        for (int i = 0; i < n; i++)
        {
            Assert.False(float.IsNaN(output[i]),      $"output[{i}] is NaN");
            Assert.False(float.IsInfinity(output[i]), $"output[{i}] is Infinity");
        }
    }

    // ── Large hidden dim (Llama-style) ────────────────────────────────────────

    [Fact]
    public void LargeHiddenDim_MatchesCpu()
    {
        const int n = 128, seqLen = 8;
        const float eps = 1e-5f;
        var rng = new Random(42);

        float[] input  = new float[seqLen * n];
        float[] weight = new float[n];
        for (int i = 0; i < input.Length; i++) input[i]  = rng.NextSingle() * 2f - 1f;
        for (int i = 0; i < n; i++)            weight[i] = rng.NextSingle() + 0.5f;

        float[] expected = CpuExpected(input, weight, n, seqLen, eps);
        float[] output   = new float[seqLen * n];

        using var ctx = new MetalContext();
        RmsNorm.Execute(ctx, input, weight, output, n, seqLen, eps);

        for (int i = 0; i < output.Length; i++) Assert.Equal(expected[i], output[i], 1e-4f);
    }

    // ── Non-default epsilon changes the result ────────────────────────────────

    [Fact]
    public void DifferentEpsilon_MatchesCpu()
    {
        const int n = 8;
        const float eps = 1e-6f; // some models (Llama 3) use 1e-5, others use 1e-6
        var rng = new Random(99);

        float[] input  = new float[n];
        float[] weight = new float[n];
        for (int i = 0; i < n; i++) { input[i] = rng.NextSingle() * 2f - 1f; weight[i] = rng.NextSingle() + 0.5f; }

        float[] expected = CpuExpected(input, weight, n, 1, eps);
        float[] output   = new float[n];

        using var ctx = new MetalContext();
        RmsNorm.Execute(ctx, input, weight, output, n, 1, eps);

        for (int i = 0; i < n; i++) Assert.Equal(expected[i], output[i], 1e-4f);
    }
}
