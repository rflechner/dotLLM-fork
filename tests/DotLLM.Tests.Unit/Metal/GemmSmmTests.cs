using DotLLM.Metal;
using DotLLM.Metal.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Metal;

// Numerical-correctness tests for the tiled simdgroup_matrix FP16 GEMM.
// The kernel only supports transposeA=false, transposeB=true, alpha=1, beta=0
// with M and N multiples of 32 and K multiple of 8 (LLM projection layout
// Y = X · Wᵀ).
public sealed class GemmF16SmmTests
{
    // FP32-accumulated CPU reference (no FP16 intermediate rounding).
    private static float[] CpuGemmTransposeB(Half[] a, Half[] b, int m, int n, int k)
    {
        float[] c = new float[m * n];
        for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
        {
            float acc = 0f;
            for (int p = 0; p < k; p++)
                acc += (float)a[i * k + p] * (float)b[j * k + p];
            c[i * n + j] = acc;
        }
        return c;
    }

    private static void AssertCloseRel(float[] expected, Half[] actual, float relTol, float floor)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i];
            float a = (float)actual[i];
            float tol = MathF.Max(MathF.Abs(e) * relTol, floor);
            Assert.True(MathF.Abs(e - a) <= tol,
                $"index {i}: expected={e}, actual={a}, diff={MathF.Abs(e - a)}, tol={tol}");
        }
    }

    // Smallest legal size — single 32×32 output tile, multiple K iterations.
    [Fact]
    public void SingleTile_32x32x32_MatchesCpu()
    {
        const int m = 32, n = 32, k = 32;
        var rng = new Random(1);

        Half[] x = new Half[m * k];
        Half[] w = new Half[n * k];
        Half[] y = new Half[m * n];
        for (int i = 0; i < x.Length; i++) x[i] = (Half)(rng.NextSingle() * 2f - 1f);
        for (int i = 0; i < w.Length; i++) w[i] = (Half)(rng.NextSingle() * 2f - 1f);

        float[] expected = CpuGemmTransposeB(x, w, m, n, k);

        using var ctx = new MetalContext();
        Gemm.ExecuteF16Smm(ctx, x, w, y, m, n, k);

        AssertCloseRel(expected, y, relTol: 5e-3f, floor: 5e-3f);
    }

    // Multi-tile output: 64×96 result needs a 3×2 threadgroup grid.
    [Fact]
    public void MultiTile_64x96_MatchesCpu()
    {
        const int m = 64, n = 96, k = 32;
        var rng = new Random(2);

        Half[] x = new Half[m * k];
        Half[] w = new Half[n * k];
        Half[] y = new Half[m * n];
        for (int i = 0; i < x.Length; i++) x[i] = (Half)(rng.NextSingle() * 2f - 1f);
        for (int i = 0; i < w.Length; i++) w[i] = (Half)(rng.NextSingle() * 2f - 1f);

        float[] expected = CpuGemmTransposeB(x, w, m, n, k);

        using var ctx = new MetalContext();
        Gemm.ExecuteF16Smm(ctx, x, w, y, m, n, k);

        AssertCloseRel(expected, y, relTol: 5e-3f, floor: 5e-3f);
    }

    // K not equal to a tile multiple of 32 — exercises the inner K-loop alone.
    // K=40 (multiple of 8 but not 32) verifies the K-iteration partial doesn't drop work.
    [Fact]
    public void K_NonMultipleOf32_MatchesCpu()
    {
        const int m = 32, n = 32, k = 40;
        var rng = new Random(11);

        Half[] x = new Half[m * k];
        Half[] w = new Half[n * k];
        Half[] y = new Half[m * n];
        for (int i = 0; i < x.Length; i++) x[i] = (Half)((rng.NextSingle() * 2f - 1f) * 0.5f);
        for (int i = 0; i < w.Length; i++) w[i] = (Half)((rng.NextSingle() * 2f - 1f) * 0.5f);

        float[] expected = CpuGemmTransposeB(x, w, m, n, k);

        using var ctx = new MetalContext();
        Gemm.ExecuteF16Smm(ctx, x, w, y, m, n, k);

        AssertCloseRel(expected, y, relTol: 5e-3f, floor: 5e-3f);
    }

    // LLM-shaped projection. Magnitudes scaled so FP32 accumulation over k=128
    // stays accurate.
    [Fact]
    public void LlamaFfnShape_MatchesCpu()
    {
        const int m = 32, k = 128, n = 256;
        var rng = new Random(42);

        Half[] x = new Half[m * k];
        Half[] w = new Half[n * k];
        Half[] y = new Half[m * n];
        for (int i = 0; i < x.Length; i++) x[i] = (Half)((rng.NextSingle() * 2f - 1f) * 0.2f);
        for (int i = 0; i < w.Length; i++) w[i] = (Half)((rng.NextSingle() * 2f - 1f) * 0.2f);

        float[] expected = CpuGemmTransposeB(x, w, m, n, k);

        using var ctx = new MetalContext();
        Gemm.ExecuteF16Smm(ctx, x, w, y, m, n, k);

        AssertCloseRel(expected, y, relTol: 1e-2f, floor: 5e-3f);
    }

    // Identity weight: Y = X · Iᵀ = X. Catches stride/transpose bugs across
    // the full 32×32 tile and the simdgroup-quadrant layout.
    [Fact]
    public void IdentityWeight_ReturnsInput()
    {
        const int m = 32, k = 32, n = 32;

        Half[] x = new Half[m * k];
        Half[] w = new Half[n * k];
        Half[] y = new Half[m * n];
        var rng = new Random(3);
        for (int i = 0; i < x.Length; i++) x[i] = (Half)(rng.NextSingle() * 2f - 1f);
        for (int i = 0; i < n; i++) w[i * k + i] = (Half)1f;

        using var ctx = new MetalContext();
        Gemm.ExecuteF16Smm(ctx, x, w, y, m, n, k);

        for (int i = 0; i < m * n; i++)
            Assert.Equal((float)x[i], (float)y[i], 1e-3f);
    }

    // Cross-check vs the MPS-backed FP16 GEMM. They should agree within FP16
    // tolerance because both accumulate in FP32 internally.
    [Fact]
    public void AgreesWithMpsGemm_Within_Fp16_Tolerance()
    {
        const int m = 64, k = 64, n = 96;
        var rng = new Random(7);

        Half[] x = new Half[m * k];
        Half[] w = new Half[n * k];
        Half[] ySmm = new Half[m * n];
        Half[] yMps = new Half[m * n];
        for (int i = 0; i < x.Length; i++) x[i] = (Half)((rng.NextSingle() * 2f - 1f) * 0.3f);
        for (int i = 0; i < w.Length; i++) w[i] = (Half)((rng.NextSingle() * 2f - 1f) * 0.3f);

        using var ctx = new MetalContext();
        Gemm.ExecuteF16Smm(ctx, x, w, ySmm, m, n, k);
        Gemm.ExecuteF16   (ctx, x, w, yMps, m, n, k, transposeB: true);

        for (int i = 0; i < m * n; i++)
        {
            float a = (float)ySmm[i];
            float b = (float)yMps[i];
            float tol = MathF.Max(MathF.Abs(b) * 1e-2f, 5e-3f);
            Assert.True(MathF.Abs(a - b) <= tol,
                $"index {i}: smm={a}, mps={b}, diff={MathF.Abs(a - b)}, tol={tol}");
        }
    }

    // ── Boundary handling: non-aligned M, N, K ───────────────────────────────

    // Realistic prefill shape: seqLen=545 (not multiple of 32), hidden=4096.
    // Down-scaled here (seqLen=37, k=4096) to keep the test fast — what
    // matters is M not being a multiple of 32.
    [Fact]
    public void M_NonMultipleOf32_MatchesCpu()
    {
        const int m = 37, k = 64, n = 96;
        var rng = new Random(101);

        Half[] x = new Half[m * k];
        Half[] w = new Half[n * k];
        Half[] y = new Half[m * n];
        for (int i = 0; i < x.Length; i++) x[i] = (Half)((rng.NextSingle() * 2f - 1f) * 0.3f);
        for (int i = 0; i < w.Length; i++) w[i] = (Half)((rng.NextSingle() * 2f - 1f) * 0.3f);

        float[] expected = CpuGemmTransposeB(x, w, m, n, k);

        using var ctx = new MetalContext();
        Gemm.ExecuteF16Smm(ctx, x, w, y, m, n, k);

        AssertCloseRel(expected, y, relTol: 1e-2f, floor: 5e-3f);
    }

    [Fact]
    public void N_NonMultipleOf32_MatchesCpu()
    {
        const int m = 32, k = 64, n = 71;
        var rng = new Random(102);

        Half[] x = new Half[m * k];
        Half[] w = new Half[n * k];
        Half[] y = new Half[m * n];
        for (int i = 0; i < x.Length; i++) x[i] = (Half)((rng.NextSingle() * 2f - 1f) * 0.3f);
        for (int i = 0; i < w.Length; i++) w[i] = (Half)((rng.NextSingle() * 2f - 1f) * 0.3f);

        float[] expected = CpuGemmTransposeB(x, w, m, n, k);

        using var ctx = new MetalContext();
        Gemm.ExecuteF16Smm(ctx, x, w, y, m, n, k);

        AssertCloseRel(expected, y, relTol: 1e-2f, floor: 5e-3f);
    }

    [Fact]
    public void K_NonMultipleOf8_MatchesCpu()
    {
        const int m = 32, k = 13, n = 32;
        var rng = new Random(103);

        Half[] x = new Half[m * k];
        Half[] w = new Half[n * k];
        Half[] y = new Half[m * n];
        for (int i = 0; i < x.Length; i++) x[i] = (Half)((rng.NextSingle() * 2f - 1f) * 0.5f);
        for (int i = 0; i < w.Length; i++) w[i] = (Half)((rng.NextSingle() * 2f - 1f) * 0.5f);

        float[] expected = CpuGemmTransposeB(x, w, m, n, k);

        using var ctx = new MetalContext();
        Gemm.ExecuteF16Smm(ctx, x, w, y, m, n, k);

        AssertCloseRel(expected, y, relTol: 5e-3f, floor: 5e-3f);
    }

    // All three dims non-aligned simultaneously.
    [Fact]
    public void All_NonAligned_MatchesCpu()
    {
        const int m = 17, k = 23, n = 41;
        var rng = new Random(104);

        Half[] x = new Half[m * k];
        Half[] w = new Half[n * k];
        Half[] y = new Half[m * n];
        for (int i = 0; i < x.Length; i++) x[i] = (Half)((rng.NextSingle() * 2f - 1f) * 0.4f);
        for (int i = 0; i < w.Length; i++) w[i] = (Half)((rng.NextSingle() * 2f - 1f) * 0.4f);

        float[] expected = CpuGemmTransposeB(x, w, m, n, k);

        using var ctx = new MetalContext();
        Gemm.ExecuteF16Smm(ctx, x, w, y, m, n, k);

        AssertCloseRel(expected, y, relTol: 1e-2f, floor: 5e-3f);
    }
}
