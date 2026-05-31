using DotLLM.Metal;
using DotLLM.Metal.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Metal;

public sealed class SiluTests
{
    [MetalTestFact]
    public void Zero_ProducesZero()
    {
        float[] input = [0f];
        float[] result = new float[1];

        using var ctx = new MetalContext();
        Silu.Execute(ctx, input, result);

        Assert.Equal(0f, result[0], 1e-6f);
    }

    [MetalTestFact]
    public void One_MatchesExpected()
    {
        // SiLU(1) = 1 * sigmoid(1) = 1 / (1 + e^-1) ≈ 0.7311
        float[] input = [1.0f];
        float[] result = new float[1];

        using var ctx = new MetalContext();
        Silu.Execute(ctx, input, result);

        Assert.Equal(0.7311f, result[0], 1e-3f);
    }

    [MetalTestFact]
    public void LargeNegative_NearZero()
    {
        // SiLU(-10) = -10 * sigmoid(-10) ≈ -10 * 0.0000454 ≈ -0.000454
        float[] input = [-10f];
        float[] result = new float[1];

        using var ctx = new MetalContext();
        Silu.Execute(ctx, input, result);

        Assert.True(MathF.Abs(result[0]) < 0.001f, $"SiLU(-10) = {result[0]}, expected near zero");
    }

    [MetalTestFact]
    public void LargePositive_ApproachesInput()
    {
        // SiLU(10) = 10 * sigmoid(10) ≈ 10 * 0.99995 ≈ 9.9995
        float[] input = [10f];
        float[] result = new float[1];

        using var ctx = new MetalContext();
        Silu.Execute(ctx, input, result);

        Assert.Equal(10f, result[0], 0.01f);
    }

    [MetalTestFact]
    public void MultipleValues_AllCorrect()
    {
        float[] input = [-5f, -1f, 0f, 1f, 5f];
        float[] result = new float[5];
        float[] expected = new float[5];

        DotLLM.Cpu.Kernels.SiLu.ExecuteScalar(input, expected);
        using var ctx = new MetalContext();
        Silu.Execute(ctx, input, result);

        for (int i = 0; i < input.Length; i++)
            Assert.Equal(expected[i], result[i], 1e-5f);
    }

    [MetalTestFact]
    public void ScalarMatchesTensorPrimitives()
    {
        var rng = new Random(42);
        const int n = 1024;
        float[] input = new float[n];
        for (int i = 0; i < n; i++)
            input[i] = rng.NextSingle() * 20f - 10f; // range [-10, 10]

        float[] scalarResult = new float[n];
        float[] simdResult = new float[n];

        DotLLM.Cpu.Kernels.SiLu.ExecuteScalar(input, scalarResult);

        using var ctx = new MetalContext();
        Silu.Execute(ctx, input, simdResult);


        for (int i = 0; i < n; i++)
            Assert.Equal(scalarResult[i], simdResult[i], 1e-5f);
    }
}
