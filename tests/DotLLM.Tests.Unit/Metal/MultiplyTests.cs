using DotLLM.Metal;
using Xunit;

namespace DotLLM.Tests.Unit.Metal;

public sealed class MultiplyTests
{
    [Fact]
    public void KnownValues_MatchExpected()
    {
        using var ctx = new MetalContext();
        float[] a = [2.0f, 3.0f, 4.0f, 5.0f];
        float[] b = [10.0f, 20.0f, 30.0f, 40.0f];
        float[] result = new float[4];

        Multiply.Execute(ctx, a, b, result);

        Assert.Equal([20.0f, 60.0f, 120.0f, 200.0f], result);
    }
}
