#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <dlfcn.h>
#include <stdint.h>
#include <string.h>
#include "dotllm_metal.h"
#include "dotllm_core.h"

// ── KV-cache (persistent MTLBuffers, zero-copy on Apple Silicon) ─────────────
//
// Each layer gets two MTLResourceStorageModeShared buffers (K and V) allocated
// once at creation. On Apple Silicon, Shared storage means the .contents pointer
// is the same physical memory the GPU reads — the C# side writes via NativeMemory.Copy
// and the kernel consumes the buffer directly, with zero copies.
//
// key_ptrs / value_ptrs are cached .contents addresses so the hot path never
// sends ObjC messages for pointer lookup.

struct dotllm_metal_kvcache {
    NSArray<id<MTLBuffer>>* key_buffers;     // [num_layers] — ARC strong
    NSArray<id<MTLBuffer>>* value_buffers;   // [num_layers] — ARC strong
    void**   key_ptrs;    // raw .contents addresses (no ObjC msg on hot path)
    void**   value_ptrs;
    int32_t  num_layers;
    int32_t  kv_stride;   // num_kv_heads * head_dim
    int32_t  max_seq_len;
    int32_t  current_length;
};

extern "C" dotllm_metal_kvcache* dotllm_metal_kvcache_create(
    dotllm_metal_context* ctx,
    int32_t num_layers,
    int32_t num_kv_heads,
    int32_t head_dim,
    int32_t max_seq_len)
{
    if (!ctx || num_layers <= 0 || num_kv_heads <= 0 || head_dim <= 0 || max_seq_len <= 0)
        return nullptr;

    NSUInteger bufBytes = (NSUInteger)((size_t)max_seq_len * num_kv_heads * head_dim * sizeof(uint16_t));

    NSMutableArray<id<MTLBuffer>>* kArr = [NSMutableArray arrayWithCapacity:num_layers];
    NSMutableArray<id<MTLBuffer>>* vArr = [NSMutableArray arrayWithCapacity:num_layers];

    void** kPtrs = (void**)malloc((size_t)num_layers * sizeof(void*));
    void** vPtrs = (void**)malloc((size_t)num_layers * sizeof(void*));
    if (!kPtrs || !vPtrs) { free(kPtrs); free(vPtrs); return nullptr; }

    for (int i = 0; i < num_layers; i++) {
        id<MTLBuffer> kb = [ctx->device newBufferWithLength:bufBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> vb = [ctx->device newBufferWithLength:bufBytes options:MTLResourceStorageModeShared];
        if (!kb || !vb) { free(kPtrs); free(vPtrs); return nullptr; }
        
        [kArr addObject:kb];
        [vArr addObject:vb];
        kPtrs[i] = kb.contents;
        vPtrs[i] = vb.contents;
    }

    auto* cache          = new dotllm_metal_kvcache();
    cache->key_buffers   = [kArr copy];
    cache->value_buffers = [vArr copy];
    cache->key_ptrs      = kPtrs;
    cache->value_ptrs    = vPtrs;
    cache->num_layers    = num_layers;
    cache->kv_stride     = num_kv_heads * head_dim;
    cache->max_seq_len   = max_seq_len;
    cache->current_length = 0;
    return cache;
}

extern "C" void dotllm_metal_kvcache_destroy(dotllm_metal_kvcache* cache)
{
    if (!cache) return;
    free(cache->key_ptrs);
    free(cache->value_ptrs);
    // ARC releases key_buffers / value_buffers (and the MTLBuffers they hold)
    // when the struct is deleted.
    delete cache;
}

extern "C" void* dotllm_metal_kvcache_key_ptr(dotllm_metal_kvcache* cache, int32_t layer)
{
    if (!cache || layer < 0 || layer >= cache->num_layers) return nullptr;
    return cache->key_ptrs[layer];
}

extern "C" void* dotllm_metal_kvcache_value_ptr(dotllm_metal_kvcache* cache, int32_t layer)
{
    if (!cache || layer < 0 || layer >= cache->num_layers) return nullptr;
    return cache->value_ptrs[layer];
}

extern "C" int32_t dotllm_metal_kvcache_current_length(dotllm_metal_kvcache* cache)
{
    return cache ? cache->current_length : 0;
}

extern "C" void dotllm_metal_kvcache_set_current_length(dotllm_metal_kvcache* cache, int32_t length)
{
    if (cache) cache->current_length = length;
}

