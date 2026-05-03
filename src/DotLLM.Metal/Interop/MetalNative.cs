using System.Runtime.InteropServices;

namespace DotLLM.Metal.Interop;

/// <summary>
/// P/Invoke declarations for the native Metal library (libdotllmmetal.dylib).
/// All kernel functions take an opaque context handle as first argument.
/// </summary>
internal static partial class MetalNative
{
    private const string LibName = "dotllmmetal";

    /// <summary>Creates a Metal context (device + command queue + pipeline cache).</summary>
    /// <returns>Opaque pointer, or <c>0</c> on failure.</returns>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_create_context")]
    internal static partial nint CreateContext();

    /// <summary>Destroys the context and releases all Metal resources.</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_destroy_context")]
    [SuppressGCTransition]
    internal static partial void DestroyContext(nint ctx);

    /// <summary>
    /// Allocates shared (CPU+GPU visible) memory and returns the .contents pointer.
    /// The backing MTLBuffer is retained by the context for zero-copy kernel access.
    /// </summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_alloc_shared")]
    internal static partial nint AllocShared(nint ctx, nuint bytes);

    /// <summary>
    /// Releases a buffer previously returned by <see cref="AllocShared"/>.
    /// </summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_free_shared")]
    [SuppressGCTransition]
    internal static partial void FreeShared(nint ctx, nint ptr);
}
