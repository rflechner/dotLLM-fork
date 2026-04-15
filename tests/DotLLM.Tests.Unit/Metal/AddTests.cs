using DotLLM.Metal;
using Xunit;

namespace DotLLM.Tests.Unit.Metal;

public sealed class AddTests
{

    [Fact]
    public void KnownValues_MatchExpected()
    {
        float[] a = [1.0f, 2.0f, 3.0f, 4.0f];
        float[] b = [10.0f, 20.0f, 30.0f, 40.0f];
        float[] result = new float[4];

        Add.Execute(a, b, result);

        Assert.Equal([11.0f, 22.0f, 33.0f, 44.0f], result);
    }

}
