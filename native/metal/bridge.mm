#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <string.h>
#include "dotllm_metal.h"

struct dotllm_metal_context {
    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;
    NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* pipelines;
};

dotllm_metal_context* dotllm_metal_create_context(void)
{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return nullptr;

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) return nullptr;

    dotllm_metal_context* ctx = new dotllm_metal_context();
    ctx->device    = device;
    ctx->queue     = queue;
    ctx->pipelines = [NSMutableDictionary new];
    return ctx;
}

void dotllm_metal_destroy_context(dotllm_metal_context* ctx)
{
    if (!ctx) return;
    // ARC releases device, queue, pipelines automatically
    delete ctx;
}

static id<MTLComputePipelineState> get_or_create_pipeline(
    dotllm_metal_context* ctx,
    const char* shaderPath,
    const char* functionName)
{
    NSString* key = [NSString stringWithFormat:@"%s|%s", shaderPath, functionName];

    id<MTLComputePipelineState> pipeline = ctx->pipelines[key];
    if (pipeline) return pipeline;

    NSError* error = nil;
    NSString* source = [NSString stringWithContentsOfFile:@(shaderPath)
                                                 encoding:NSUTF8StringEncoding
                                                    error:&error];
    if (!source) return nil;

    id<MTLLibrary> library = [ctx->device newLibraryWithSource:source options:nil error:&error];
    if (!library) return nil;

    id<MTLFunction> function = [library newFunctionWithName:@(functionName)];
    if (!function) return nil;

    pipeline = [ctx->device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) return nil;

    ctx->pipelines[key] = pipeline;
    return pipeline;
}

static int run_binary_f32_kernel(
    dotllm_metal_context* ctx,
    const char* shaderPath,
    const char* functionName,
    const float* a,
    const float* b,
    float* result,
    uint32_t length)
{
    @autoreleasepool {
        if (!ctx || !a || !b || !result) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, shaderPath, functionName);
        if (!pipeline) return -3;

        NSUInteger bytes = (NSUInteger)length * sizeof(float);
        id<MTLBuffer> bufA = [ctx->device newBufferWithBytes:a length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [ctx->device newBufferWithBytes:b length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufR = [ctx->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufL = [ctx->device newBufferWithBytes:&length length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        if (!bufA || !bufB || !bufR || !bufL) return -7;

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        if (!cmd) return -8;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufA offset:0 atIndex:0];
        [enc setBuffer:bufB offset:0 atIndex:1];
        [enc setBuffer:bufR offset:0 atIndex:2];
        [enc setBuffer:bufL offset:0 atIndex:3];

        NSUInteger tgw = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
        if (length > 0 && tgw > length) tgw = length;
        if (tgw == 0) tgw = 1;

        [enc dispatchThreads:MTLSizeMake(length, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.error != nil) return -11;

        memcpy(result, bufR.contents, bytes);
        return 0;
    }
}

static int run_unary_f32_kernel(
    dotllm_metal_context* ctx,
    const char* shaderPath,
    const char* functionName,
    const float* a,
    float* result,
    uint32_t length)
{
    @autoreleasepool {
        if (!ctx || !a || !result) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, shaderPath, functionName);
        if (!pipeline) return -3;

        NSUInteger bytes = (NSUInteger)length * sizeof(float);
        id<MTLBuffer> bufA = [ctx->device newBufferWithBytes:a length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufR = [ctx->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufL = [ctx->device newBufferWithBytes:&length length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        if (!bufA || !bufR || !bufL) return -7;

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        if (!cmd) return -8;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufA offset:0 atIndex:0];
        [enc setBuffer:bufR offset:0 atIndex:2];
        [enc setBuffer:bufL offset:0 atIndex:3];

        NSUInteger tgw = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
        if (length > 0 && tgw > length) tgw = length;
        if (tgw == 0) tgw = 1;

        [enc dispatchThreads:MTLSizeMake(length, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.error != nil) return -11;

        memcpy(result, bufR.contents, bytes);
        return 0;
    }
}

extern "C" int dotllm_metal_add_f32(
    dotllm_metal_context* ctx,
    const float* a,
    const float* b,
    float* result,
    uint32_t length)
{
    return run_binary_f32_kernel(ctx, "add.metal", "add_arrays", a, b, result, length);
}

extern "C" int dotllm_metal_multiply_f32(
    dotllm_metal_context* ctx,
    const float* a,
    const float* b,
    float* result,
    uint32_t length)
{
    return run_binary_f32_kernel(ctx, "multiply.metal", "multiply_arrays", a, b, result, length);
}

extern "C" int dotllm_metal_softmax_f32(
    dotllm_metal_context* ctx,
    const float* input,
    float* result,
    uint32_t length)
{
    return run_unary_f32_kernel(ctx, "softmax.metal", "softmax", input, result, length);
}

extern "C" int dotllm_metal_silu_f32(
    dotllm_metal_context* ctx,
    const float* input,
    float* result,
    uint32_t length)
{
    return run_unary_f32_kernel(ctx, "silu.metal", "silu", input, result, length);
}

extern "C" int dotllm_metal_swiglu_f32(
    dotllm_metal_context* ctx,
    const float* gate,
    const float* up,
    float* result,
    uint32_t length)
{
    return run_binary_f32_kernel(ctx, "swiglu.metal", "swiglu", gate, up, result, length);
}

extern "C" int dotllm_metal_bias_add_f32(
    dotllm_metal_context* ctx,
    float* output,
    const float* bias,
    uint32_t dim,
    uint32_t seq_len)
{
    @autoreleasepool {
        if (!ctx || !output || !bias) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "bias_add.metal", "bias_add");
        if (!pipeline) return -3;

        uint32_t total = dim * seq_len;
        NSUInteger outputBytes = (NSUInteger)total * sizeof(float);
        NSUInteger biasBytes   = (NSUInteger)dim  * sizeof(float);

        // output is in-place: wrap the existing host buffer (no copy in)
        id<MTLBuffer> bufOutput = [ctx->device newBufferWithBytesNoCopy:output
                                                                  length:outputBytes
                                                                 options:MTLResourceStorageModeShared
                                                             deallocator:nil];
        id<MTLBuffer> bufBias   = [ctx->device newBufferWithBytes:bias
                                                           length:biasBytes
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDim    = [ctx->device newBufferWithBytes:&dim
                                                           length:sizeof(uint32_t)
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufSeqLen = [ctx->device newBufferWithBytes:&seq_len
                                                           length:sizeof(uint32_t)
                                                          options:MTLResourceStorageModeShared];
        if (!bufOutput || !bufBias || !bufDim || !bufSeqLen) return -7;

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        if (!cmd) return -8;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufOutput offset:0 atIndex:0];
        [enc setBuffer:bufBias   offset:0 atIndex:1];
        [enc setBuffer:bufDim    offset:0 atIndex:2];
        [enc setBuffer:bufSeqLen offset:0 atIndex:3];

        NSUInteger tgw = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
        if (total > 0 && tgw > total) tgw = total;
        if (tgw == 0) tgw = 1;

        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.error != nil) return -11;

        // output buffer wraps host memory directly — no memcpy needed
        return 0;
    }
}

extern "C" int dotllm_metal_rope_f32(
    dotllm_metal_context* ctx,
    float*         q,
    float*         k,
    const int32_t* positions,
    int32_t        seq_len,
    int32_t        num_heads,
    int32_t        num_kv_heads,
    int32_t        head_dim,
    int32_t        rope_dim,
    float          theta,
    int32_t        rope_type)   // 0 = norm, 1 = neox
{
    @autoreleasepool {
        if (!ctx || !q || !k || !positions) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "RoPE.metal", "rope_f32");
        if (!pipeline) return -3;

        int32_t half_rope     = rope_dim / 2;
        uint32_t total_q      = (uint32_t)(seq_len * num_heads    * half_rope);
        uint32_t total_k      = (uint32_t)(seq_len * num_kv_heads * half_rope);
        uint32_t dispatch_len = total_q > total_k ? total_q : total_k;

        NSUInteger qBytes   = (NSUInteger)(seq_len * num_heads    * head_dim) * sizeof(float);
        NSUInteger kBytes   = (NSUInteger)(seq_len * num_kv_heads * head_dim) * sizeof(float);
        NSUInteger posBytes = (NSUInteger)seq_len * sizeof(int32_t);

        id<MTLBuffer> bufQ    = [ctx->device newBufferWithBytesNoCopy:q length:qBytes
                                    options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> bufK    = [ctx->device newBufferWithBytesNoCopy:k length:kBytes
                                    options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> bufPos  = [ctx->device newBufferWithBytes:positions length:posBytes
                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufSL   = [ctx->device newBufferWithBytes:&seq_len      length:sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufNH   = [ctx->device newBufferWithBytes:&num_heads    length:sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufNKV  = [ctx->device newBufferWithBytes:&num_kv_heads length:sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufHD   = [ctx->device newBufferWithBytes:&head_dim     length:sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufRD   = [ctx->device newBufferWithBytes:&rope_dim     length:sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufTh   = [ctx->device newBufferWithBytes:&theta        length:sizeof(float)   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufRT   = [ctx->device newBufferWithBytes:&rope_type    length:sizeof(int32_t) options:MTLResourceStorageModeShared];

        if (!bufQ || !bufK || !bufPos || !bufSL || !bufNH ||
            !bufNKV || !bufHD || !bufRD || !bufTh || !bufRT) return -7;

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        if (!cmd) return -8;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufQ   offset:0 atIndex:0];
        [enc setBuffer:bufK   offset:0 atIndex:1];
        [enc setBuffer:bufPos offset:0 atIndex:2];
        [enc setBuffer:bufSL  offset:0 atIndex:3];
        [enc setBuffer:bufNH  offset:0 atIndex:4];
        [enc setBuffer:bufNKV offset:0 atIndex:5];
        [enc setBuffer:bufHD  offset:0 atIndex:6];
        [enc setBuffer:bufRD  offset:0 atIndex:7];
        [enc setBuffer:bufTh  offset:0 atIndex:8];
        [enc setBuffer:bufRT  offset:0 atIndex:9];

        NSUInteger tgw = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
        if (dispatch_len > 0 && tgw > dispatch_len) tgw = dispatch_len;
        if (tgw == 0) tgw = 1;

        [enc dispatchThreads:MTLSizeMake(dispatch_len, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.error != nil) return -11;
        return 0;
    }
}

extern "C" int dotllm_metal_rmsnorm_f32(
    dotllm_metal_context* ctx,
    const float* input,
    const float* weight,
    float*       output,
    int32_t      n,
    int32_t      seq_len,
    float        eps)
{
    @autoreleasepool {
        if (!ctx || !input || !weight || !output) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "rmsnorm.metal", "rmsnorm_f32");
        if (!pipeline) return -3;

        // Threadgroup size: up to 256 threads per row.
        // This kernel uses dispatchThreadgroups (not dispatchThreads) because
        // the unit of work here is a whole group, not a single thread.
        uint32_t tgSize = (uint32_t)MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);

        NSUInteger inputBytes  = (NSUInteger)(seq_len * n) * sizeof(float);
        NSUInteger weightBytes = (NSUInteger)n * sizeof(float);

        id<MTLBuffer> bufIn  = [ctx->device newBufferWithBytes:input  length:inputBytes  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufW   = [ctx->device newBufferWithBytes:weight length:weightBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufOut = [ctx->device newBufferWithLength:inputBytes               options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufN   = [ctx->device newBufferWithBytes:&n   length:sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufEps = [ctx->device newBufferWithBytes:&eps length:sizeof(float)   options:MTLResourceStorageModeShared];

        if (!bufIn || !bufW || !bufOut || !bufN || !bufEps) return -7;

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        if (!cmd) return -8;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufIn  offset:0 atIndex:0];
        [enc setBuffer:bufW   offset:0 atIndex:1];
        [enc setBuffer:bufOut offset:0 atIndex:2];
        [enc setBuffer:bufN   offset:0 atIndex:3];
        [enc setBuffer:bufEps offset:0 atIndex:4];

        // dispatchThreadgroups: launches exactly seq_len groups.
        // Each group = one token; tgSize threads collaborate on n elements.
        // Unlike dispatchThreads, this gives explicit control over group count.
        [enc dispatchThreadgroups:MTLSizeMake(seq_len, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.error != nil) return -11;

        memcpy(output, bufOut.contents, inputBytes);
        return 0;
    }
}
