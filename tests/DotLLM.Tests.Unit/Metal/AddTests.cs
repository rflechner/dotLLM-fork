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

    [Fact]
    public void EmptyArrays_ReturnsSilently()
    {
        float[] a = [];
        float[] b = [];
        float[] result = [];

        Add.Execute(a, b, result);

        Assert.Empty(result);
    }

    [Fact]
    public void InputLengthMismatch_ThrowsArgumentException()
    {
        float[] a = [1.0f, 2.0f];
        float[] b = [1.0f];
        float[] result = new float[2];

        Assert.Throws<ArgumentException>(() => Add.Execute(a, b, result));
    }

    [Fact]
    public void ResultTooSmall_ThrowsArgumentException()
    {
        float[] a = [1.0f, 2.0f];
        float[] b = [3.0f, 4.0f];
        float[] result = new float[1];

        Assert.Throws<ArgumentException>(() => Add.Execute(a, b, result));
    }

    [Fact]
    public void LargeVector_MatchExpected()
    {
        const int size = 1024 * 1024;
        float[] a = Enumerable.Repeat(1.0f, size).ToArray();
        float[] b = Enumerable.Repeat(2.0f, size).ToArray();
        float[] result = new float[size];

        Add.Execute(a, b, result);

        Assert.All(result, x => Assert.Equal(3.0f, x));
    }
}
