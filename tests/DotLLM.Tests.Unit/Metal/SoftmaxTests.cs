using DotLLM.Metal;
using Xunit;

namespace DotLLM.Tests.Unit.Metal;

public class SoftmaxTests
{
    [Fact]
    public void Uniform_ProducesEqual()
    {
        const int n = 4;
        float[] input = [1f, 1f, 1f, 1f];
        float[] result = new float[n];

        using var ctx = new MetalContext();
        Softmax.Execute(ctx, input, result);

        float expected = 1.0f / n;
        for (int i = 0; i < n; i++)
            Assert.Equal(expected, result[i], 1e-5f);
    }

    [Fact]
    public void SumsToOne()
    {
        float[] input = [1f, 2f, 3f, 4f, 5f];
        float[] result = new float[5];

        using var ctx = new MetalContext();
        Softmax.Execute(ctx, input, result);

        float sum = 0;
        for (int i = 0; i < result.Length; i++)
            sum += result[i];

        Assert.Equal(1.0f, sum, 1e-5f);
    }

    [Fact]
    public void LargeValues_Stable()
    {
        // Should not overflow — numerically stable softmax subtracts max.
        float[] input = [1000f, 1001f, 1002f];
        float[] result = new float[3];

        using var ctx = new MetalContext();
        Softmax.Execute(ctx, input, result);

        Assert.All(result, v => Assert.True(float.IsFinite(v), $"Non-finite value: {v}"));

        float sum = 0;
        for (int i = 0; i < result.Length; i++)
            sum += result[i];
        Assert.Equal(1.0f, sum, 1e-5f);

        // Largest input should get largest probability.
        Assert.True(result[2] > result[1]);
        Assert.True(result[1] > result[0]);
    }

    [Fact]
    public void SingleElement_ProducesOne()
    {
        float[] input = [42f];
        float[] result = new float[1];

        using var ctx = new MetalContext();
        Softmax.Execute(ctx, input, result);

        Assert.Equal(1.0f, result[0], 1e-5f);
    }

    [Fact]
    public void AllNegative_StillValid()
    {
        float[] input = [-10f, -20f, -30f];
        float[] result = new float[3];

        using var ctx = new MetalContext();
        Softmax.Execute(ctx, input, result);

        Assert.All(result, v => Assert.True(v > 0f, $"Expected positive, got {v}"));

        float sum = 0;
        for (int i = 0; i < result.Length; i++)
            sum += result[i];
        Assert.Equal(1.0f, sum, 1e-5f);
    }
}
