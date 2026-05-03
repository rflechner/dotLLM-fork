#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dotllm_metal_context dotllm_metal_context;

/// Opaque KV-cache backed by persistent MTLResourceStorageModeShared buffers.
/// On Apple Silicon, Shared storage is physically accessible by both the CPU
/// and the GPU — the C# side writes K/V via the contents pointer and the
/// attention kernel reads directly, with zero copies.
typedef struct dotllm_metal_kvcache dotllm_metal_kvcache;

struct dotllm_metal_context {
    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary>      library;       // pre-compiled dotllm_kernels.metallib, loaded once
    NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* pipelines;
};


#ifdef __cplusplus
}
#endif
