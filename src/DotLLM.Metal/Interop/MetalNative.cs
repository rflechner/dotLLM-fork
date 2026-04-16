using System.Runtime.InteropServices;

namespace DotLLM.Metal.Interop;

/// <summary>
/// P/Invoke declarations for the native Metal library (libdotllmmetal.dylib).
/// All kernel functions take an opaque context handle as first argument.
/// </summary>
internal static partial class MetalNative
{
    private const string LibName = "dotllmmetal";

    // ── Context ──────────────────────────────────────────────────────────

    /// <summary>Creates a Metal context (device + command queue + pipeline cache).</summary>
    /// <returns>Opaque pointer, or <c>0</c> on failure.</returns>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_create_context")]
    internal static partial nint CreateContext();

    /// <summary>Destroys the context and releases all Metal resources.</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_destroy_context")]
    [SuppressGCTransition]
    internal static partial void DestroyContext(nint ctx);

    // ── Kernels ───────────────────────────────────────────────────────────

    /// <summary>Element-wise addition: result[i] = a[i] + b[i]</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_add_f32")]
    internal static unsafe partial int AddF32(
        nint ctx,
        float* a,
        float* b,
        float* result,
        uint length);

    /// <summary>Element-wise multiplication: result[i] = a[i] * b[i]</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_multiply_f32")]
    internal static unsafe partial int MultiplyF32(
        nint ctx,
        float* a,
        float* b,
        float* result,
        uint length);
}
