using DotLLM.Metal;
using DotLLM.Models.Architectures;
using Xunit;

namespace DotLLM.Tests.Unit.Metal;

public sealed class BiasAddTests
{
    [Fact]
    public void SingleToken_BiasIsAdded()
    {
        // seqLen=1, dim=4 — trivial case: output[0, i] += bias[i]
        float[] output = [1.0f, 2.0f, 3.0f, 4.0f];
        float[] bias   = [10.0f, 20.0f, 30.0f, 40.0f];

        using var ctx = new MetalContext();
        BiasAdd.Execute(ctx, output, bias, dim: 4, seqLen: 1);

        Assert.Equal([11.0f, 22.0f, 33.0f, 44.0f], output);
    }

    [Fact]
    public void MultipleTokens_SameBiasAppliedToEach()
    {
        // seqLen=3, dim=2
        // token 0: [1, 2], token 1: [3, 4], token 2: [5, 6]
        // bias: [10, 20]
        float[] output = [1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f];
        float[] bias   = [10.0f, 20.0f];

        using var ctx = new MetalContext();
        BiasAdd.Execute(ctx, output, bias, dim: 2, seqLen: 3);

        Assert.Equal([11.0f, 22.0f, 13.0f, 24.0f, 15.0f, 26.0f], output);
    }

    [Fact]
    public void ZeroBias_OutputUnchanged()
    {
        float[] output   = [1.0f, 2.0f, 3.0f, 4.0f];
        float[] bias     = [0.0f, 0.0f, 0.0f, 0.0f];
        float[] expected = [1.0f, 2.0f, 3.0f, 4.0f];

        using var ctx = new MetalContext();
        BiasAdd.Execute(ctx, output, bias, dim: 4, seqLen: 1);

        Assert.Equal(expected, output);
    }

    [Fact]
    public void NegativeBias_SubtractsCorrectly()
    {
        float[] output = [5.0f, 5.0f, 5.0f];
        float[] bias   = [-1.0f, -2.0f, -3.0f];

        using var ctx = new MetalContext();
        BiasAdd.Execute(ctx, output, bias, dim: 3, seqLen: 1);

        Assert.Equal([4.0f, 3.0f, 2.0f], output);
    }

    [Fact]
    public void ScalarReference_MatchesMetal()
    {
        // Verifies Metal output matches a CPU scalar reference on random data.
        var rng = new Random(42);
        const int dim = 512;
        const int seqLen = 8;

        float[] output   = new float[dim * seqLen];
        float[] bias     = new float[dim];
        float[] expected = new float[dim * seqLen];

        for (int i = 0; i < output.Length; i++) output[i] = rng.NextSingle() * 10f - 5f;
        for (int i = 0; i < bias.Length;   i++) bias[i]   = rng.NextSingle() * 2f  - 1f;

        // CPU scalar reference — set expected value from CPU backend
        Array.Copy(output, expected, output.Length);
        TransformerModel.AddBias(bias, expected, dim, seqLen);

        using var ctx = new MetalContext();
        BiasAdd.Execute(ctx, output, bias, dim, seqLen);

        // Verify Metal output matches CPU reference
        for (int i = 0; i < output.Length; i++)
            Assert.Equal(expected[i], output[i], 1e-5f);
    }

    [Fact]
    public void BiasMismatch_ThrowsArgumentException()
    {
        using var ctx = new MetalContext();
        float[] output = new float[8];
        float[] bias   = new float[3];

        Assert.Throws<ArgumentException>(() => BiasAdd.Execute(ctx, output, bias, dim: 4, seqLen: 2));
    }

    [Fact]
    public void OutputSizeMismatch_ThrowsArgumentException()
    {
        using var ctx = new MetalContext();
        float[] output = new float[7];
        float[] bias   = new float[4];

        Assert.Throws<ArgumentException>(() => BiasAdd.Execute(ctx, output, bias, dim: 4, seqLen: 2));
    }

    [Fact]
    public void EmptyOutput_ReturnsSilently()
    {
        using var ctx = new MetalContext();
        float[] output = [];
        float[] bias   = [];

        BiasAdd.Execute(ctx, output, bias, dim: 0, seqLen: 0);

        Assert.Empty(output);
    }
}
