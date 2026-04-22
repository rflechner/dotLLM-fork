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
    return run_binary_f32_kernel(ctx, "add.metal", "add_f32", a, b, result, length);
}

extern "C" int dotllm_metal_add_f16(
    dotllm_metal_context* ctx,
    const uint16_t* a,
    const uint16_t* b,
    uint16_t* result,
    uint32_t length)
{
    @autoreleasepool {
        if (!ctx || !a || !b || !result) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "add.metal", "add_f16");
        if (!pipeline) return -3;

        NSUInteger bytes = (NSUInteger)length * sizeof(uint16_t);
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

        // Dispatch n/2 threads (vectorized half2); +1 handles the odd-tail guard
        uint32_t dispatch = (length / 2u) + 1u;
        NSUInteger tgw = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256u);
        if (dispatch > 0 && tgw > dispatch) tgw = dispatch;
        if (tgw == 0) tgw = 1;

        [enc dispatchThreads:MTLSizeMake(dispatch, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.error != nil) return -11;

        memcpy(result, bufR.contents, bytes);
        return 0;
    }
}

extern "C" int dotllm_metal_add_f32_f16(
    dotllm_metal_context* ctx,
    const float*    a,
    const uint16_t* b,
    float*          result,
    uint32_t        length)
{
    @autoreleasepool {
        if (!ctx || !a || !b || !result) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "add.metal", "add_f32_f16");
        if (!pipeline) return -3;

        NSUInteger bytesF32 = (NSUInteger)length * sizeof(float);
        NSUInteger bytesF16 = (NSUInteger)length * sizeof(uint16_t);
        id<MTLBuffer> bufA = [ctx->device newBufferWithBytes:a length:bytesF32 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [ctx->device newBufferWithBytes:b length:bytesF16 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufR = [ctx->device newBufferWithLength:bytesF32 options:MTLResourceStorageModeShared];
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

        NSUInteger tgw = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256u);
        if (length > 0 && tgw > length) tgw = length;
        if (tgw == 0) tgw = 1;

        [enc dispatchThreads:MTLSizeMake(length, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.error != nil) return -11;

        memcpy(result, bufR.contents, bytesF32);
        return 0;
    }
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
    float*          output,
    const uint16_t* bias,       // FP16 bias — matches bias_add_f32.cu
    uint32_t        dim,
    uint32_t        seq_len)
{
    @autoreleasepool {
        if (!ctx || !output || !bias) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "bias_add.metal", "bias_add_f32");
        if (!pipeline) return -3;

        uint32_t total = dim * seq_len;
        NSUInteger outputBytes = (NSUInteger)total * sizeof(float);
        NSUInteger biasBytes   = (NSUInteger)dim   * sizeof(uint16_t);

        // output is in-place: copy in, process, copy out
        id<MTLBuffer> bufOutput = [ctx->device newBufferWithBytes:output
                                                           length:outputBytes
                                                          options:MTLResourceStorageModeShared];
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

        memcpy(output, bufOutput.contents, outputBytes);
        return 0;
    }
}

extern "C" int dotllm_metal_bias_add_f16(
    dotllm_metal_context* ctx,
    uint16_t*       output,     // FP16 in-place
    const uint16_t* bias,       // FP16 bias
    uint32_t        dim,
    uint32_t        seq_len)
{
    @autoreleasepool {
        if (!ctx || !output || !bias) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "bias_add.metal", "bias_add_f16");
        if (!pipeline) return -3;

        uint32_t total = dim * seq_len;
        NSUInteger outputBytes = (NSUInteger)total * sizeof(uint16_t);
        NSUInteger biasBytes   = (NSUInteger)dim   * sizeof(uint16_t);

        id<MTLBuffer> bufOutput = [ctx->device newBufferWithBytes:output
                                                           length:outputBytes
                                                          options:MTLResourceStorageModeShared];
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

        // Dispatch total/2 + 1 threads — vectorized half2 path + odd-tail guard
        uint32_t dispatch = (total / 2u) + 1u;
        NSUInteger tgw = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256u);
        if (dispatch > 0 && tgw > dispatch) tgw = dispatch;
        if (tgw == 0) tgw = 1;

        [enc dispatchThreads:MTLSizeMake(dispatch, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.error != nil) return -11;

        memcpy(output, bufOutput.contents, outputBytes);
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

// Helper shared by both convert kernels.
// srcElemSize / dstElemSize: sizeof(half)=2, sizeof(float)=4.
static int run_convert_kernel(
    dotllm_metal_context* ctx,
    const char* functionName,
    const void* src,
    void*       dst,
    int32_t     n,
    size_t      srcElemSize,
    size_t      dstElemSize)
{
    @autoreleasepool {
        if (!ctx || !src || !dst || n <= 0) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "convert.metal", functionName);
        if (!pipeline) return -3;

        NSUInteger srcBytes = (NSUInteger)n * srcElemSize;
        NSUInteger dstBytes = (NSUInteger)n * dstElemSize;

        id<MTLBuffer> bufSrc = [ctx->device newBufferWithBytes:src length:srcBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDst = [ctx->device newBufferWithLength:dstBytes           options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufN   = [ctx->device newBufferWithBytes:&n length:sizeof(int32_t) options:MTLResourceStorageModeShared];
        if (!bufSrc || !bufDst || !bufN) return -7;

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        if (!cmd) return -8;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufSrc offset:0 atIndex:0];
        [enc setBuffer:bufDst offset:0 atIndex:1];
        [enc setBuffer:bufN   offset:0 atIndex:2];

        NSUInteger tgw = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
        if ((NSUInteger)n < tgw) tgw = (NSUInteger)n;

        [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.error != nil) return -11;

        memcpy(dst, bufDst.contents, dstBytes);
        return 0;
    }
}

extern "C" int dotllm_metal_per_head_rmsnorm_f32(
    dotllm_metal_context* ctx,
    float*       qk,
    const float* weight,
    int32_t      num_heads,
    int32_t      head_dim,
    int32_t      seq_len,
    float        eps)
{
    @autoreleasepool {
        if (!ctx || !qk || !weight) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "per_head_rmsnorm_f32.metal", "per_head_rmsnorm_f32");
        if (!pipeline) return -3;

        uint32_t tgSize   = (uint32_t)MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
        uint32_t nGroups  = (uint32_t)(seq_len * num_heads); // one threadgroup per (token, head)

        NSUInteger qkBytes     = (NSUInteger)(seq_len * num_heads * head_dim) * sizeof(float);
        NSUInteger weightBytes = (NSUInteger)head_dim * sizeof(float);

        // qk is in-place: wrap host memory directly, no copy-in needed.
        id<MTLBuffer> bufQK     = [ctx->device newBufferWithBytesNoCopy:qk length:qkBytes
                                      options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> bufWeight = [ctx->device newBufferWithBytes:weight length:weightBytes
                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufEps    = [ctx->device newBufferWithBytes:&eps      length:sizeof(float)   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufNH     = [ctx->device newBufferWithBytes:&num_heads length:sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufHD     = [ctx->device newBufferWithBytes:&head_dim  length:sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufSL     = [ctx->device newBufferWithBytes:&seq_len   length:sizeof(int32_t) options:MTLResourceStorageModeShared];

        if (!bufQK || !bufWeight || !bufEps || !bufNH || !bufHD || !bufSL) return -7;

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        if (!cmd) return -8;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufQK     offset:0 atIndex:0];
        [enc setBuffer:bufWeight offset:0 atIndex:1];
        [enc setBuffer:bufEps    offset:0 atIndex:2];
        [enc setBuffer:bufNH     offset:0 atIndex:3];
        [enc setBuffer:bufHD     offset:0 atIndex:4];
        [enc setBuffer:bufSL     offset:0 atIndex:5];

        // One threadgroup per (token, head) pair.
        [enc dispatchThreadgroups:MTLSizeMake(nGroups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.error != nil) return -11;
        // bufQK wraps host memory directly — no memcpy needed.
        return 0;
    }
}

extern "C" int dotllm_metal_fused_add_rmsnorm_f16(
    dotllm_metal_context* ctx,
    uint16_t*       residual,
    const uint16_t* x,
    const uint16_t* weight,
    uint16_t*       output,
    int32_t         n,
    int32_t         seq_len,
    float           eps)
{
    @autoreleasepool {
        if (!ctx || !residual || !x || !weight || !output) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "fused_add_rmsnorm.metal", "fused_add_rmsnorm_f16");
        if (!pipeline) return -3;

        uint32_t tgSize = (uint32_t)MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);

        NSUInteger rowBytes    = (NSUInteger)n * sizeof(uint16_t);
        NSUInteger totalBytes  = (NSUInteger)(seq_len * n) * sizeof(uint16_t);

        // residual is read AND written by the kernel — copy in, copy out.
        id<MTLBuffer> bufRes    = [ctx->device newBufferWithBytes:residual length:totalBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufX      = [ctx->device newBufferWithBytes:x        length:totalBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufW      = [ctx->device newBufferWithBytes:weight    length:rowBytes   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufOut    = [ctx->device newBufferWithLength:totalBytes                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufN      = [ctx->device newBufferWithBytes:&n       length:sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufEps    = [ctx->device newBufferWithBytes:&eps     length:sizeof(float)   options:MTLResourceStorageModeShared];

        if (!bufRes || !bufX || !bufW || !bufOut || !bufN || !bufEps) return -7;

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        if (!cmd) return -8;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufRes  offset:0 atIndex:0];
        [enc setBuffer:bufX    offset:0 atIndex:1];
        [enc setBuffer:bufW    offset:0 atIndex:2];
        [enc setBuffer:bufOut  offset:0 atIndex:3];
        [enc setBuffer:bufN    offset:0 atIndex:4];
        [enc setBuffer:bufEps  offset:0 atIndex:5];

        // One threadgroup per token.
        [enc dispatchThreadgroups:MTLSizeMake(seq_len, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.error != nil) return -11;

        // residual was updated in-place by the kernel — write back to host.
        memcpy(residual, bufRes.contents, totalBytes);
        memcpy(output,   bufOut.contents, totalBytes);
        return 0;
    }
}

// ── Embedding lookup helper ───────────────────────────────────────────────────
// All three variants share the same buffer layout and dispatch pattern.
// embed_table_bytes: total byte size of the embed table (varies by dtype).
static int run_embedding_kernel(
    dotllm_metal_context* ctx,
    const char*  functionName,
    const void*  embed_table,
    NSUInteger   embed_table_bytes,
    const int32_t* token_ids,
    float*       output,
    int32_t      hidden_size,
    int32_t      seq_len)
{
    @autoreleasepool {
        if (!ctx || !embed_table || !token_ids || !output) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "embedding_f32out.metal", functionName);
        if (!pipeline) return -3;

        uint32_t tgSize     = (uint32_t)MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
        NSUInteger posBytes = (NSUInteger)seq_len * sizeof(int32_t);
        NSUInteger outBytes = (NSUInteger)(seq_len * hidden_size) * sizeof(float);

        id<MTLBuffer> bufTable  = [ctx->device newBufferWithBytes:embed_table length:embed_table_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufIds    = [ctx->device newBufferWithBytes:token_ids   length:posBytes          options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufOut    = [ctx->device newBufferWithLength:outBytes                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufSL     = [ctx->device newBufferWithBytes:&seq_len    length:sizeof(int32_t)   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufHS     = [ctx->device newBufferWithBytes:&hidden_size length:sizeof(int32_t)  options:MTLResourceStorageModeShared];

        if (!bufTable || !bufIds || !bufOut || !bufSL || !bufHS) return -7;

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        if (!cmd) return -8;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufTable offset:0 atIndex:0];
        [enc setBuffer:bufIds   offset:0 atIndex:1];
        [enc setBuffer:bufOut   offset:0 atIndex:2];
        [enc setBuffer:bufSL    offset:0 atIndex:3];
        [enc setBuffer:bufHS    offset:0 atIndex:4];

        // One threadgroup per token; threads copy/dequantize hidden_size elements.
        [enc dispatchThreadgroups:MTLSizeMake(seq_len, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.error != nil) return -11;

        memcpy(output, bufOut.contents, outBytes);
        return 0;
    }
}

extern "C" int dotllm_metal_embedding_f32_f32out(
    dotllm_metal_context* ctx,
    const float*   embed_table,
    const int32_t* token_ids,
    float*         output,
    int32_t        vocab_size,
    int32_t        hidden_size,
    int32_t        seq_len)
{
    NSUInteger tableBytes = (NSUInteger)(vocab_size * hidden_size) * sizeof(float);
    return run_embedding_kernel(ctx, "embedding_lookup_f32_f32out",
        embed_table, tableBytes, token_ids, output, hidden_size, seq_len);
}

extern "C" int dotllm_metal_embedding_f16_f32out(
    dotllm_metal_context* ctx,
    const uint16_t* embed_table,
    const int32_t*  token_ids,
    float*          output,
    int32_t         vocab_size,
    int32_t         hidden_size,
    int32_t         seq_len)
{
    NSUInteger tableBytes = (NSUInteger)(vocab_size * hidden_size) * sizeof(uint16_t);
    return run_embedding_kernel(ctx, "embedding_lookup_f16_f32out",
        embed_table, tableBytes, token_ids, output, hidden_size, seq_len);
}

extern "C" int dotllm_metal_embedding_q8_0_f32out(
    dotllm_metal_context* ctx,
    const uint8_t* embed_table,
    const int32_t* token_ids,
    float*         output,
    int32_t        vocab_size,
    int32_t        hidden_size,
    int32_t        seq_len)
{
    const int32_t Q8_0_BLOCK_SIZE  = 32;
    const int32_t Q8_0_BLOCK_BYTES = 34;
    int32_t blocks_per_row = hidden_size / Q8_0_BLOCK_SIZE;
    NSUInteger tableBytes = (NSUInteger)(vocab_size * blocks_per_row) * Q8_0_BLOCK_BYTES;
    return run_embedding_kernel(ctx, "embedding_lookup_q8_0_f32out",
        embed_table, tableBytes, token_ids, output, hidden_size, seq_len);
}

// ── Attention ────────────────────────────────────────────────────────────────
//
// Shared dispatch helper: both f16 and f32 variants use the same layout,
// the only difference is element size and shader name.
//
static int run_attention_kernel(
    dotllm_metal_context* ctx,
    const char*  functionName,
    const void*  q,
    const void*  k,
    const void*  v,
    void*        output,
    size_t       elemSize,   // sizeof(float) or sizeof(uint16_t)
    int32_t      seq_q,
    int32_t      seq_kv,
    int32_t      num_heads,
    int32_t      num_kv_heads,
    int32_t      head_dim,
    int32_t      position_offset,
    int32_t      sliding_window)
{
    @autoreleasepool {
        if (!ctx || !q || !k || !v || !output) return -10;
        if (seq_q <= 0 || seq_kv <= 0 || num_heads <= 0 || num_kv_heads <= 0
            || head_dim <= 0 || num_heads % num_kv_heads != 0) return -10;

        NSString* shaderFile = [NSString stringWithFormat:@"attention_%s.metal",
                                (elemSize == sizeof(float)) ? "f32" : "f16"];
        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, shaderFile.UTF8String, functionName);
        if (!pipeline) return -3;

        NSUInteger qBytes      = (NSUInteger)(seq_q  * num_heads    * head_dim) * elemSize;
        NSUInteger kvBytes     = (NSUInteger)(seq_kv * num_kv_heads * head_dim) * elemSize;
        NSUInteger outputBytes = qBytes;

        id<MTLBuffer> bufQ   = [ctx->device newBufferWithBytes:q      length:qBytes      options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufK   = [ctx->device newBufferWithBytes:k      length:kvBytes     options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufV   = [ctx->device newBufferWithBytes:v      length:kvBytes     options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufOut = [ctx->device newBufferWithLength:outputBytes               options:MTLResourceStorageModeShared];

        id<MTLBuffer> bufSeqQ    = [ctx->device newBufferWithBytes:&seq_q           length:sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufSeqKV   = [ctx->device newBufferWithBytes:&seq_kv          length:sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufNH      = [ctx->device newBufferWithBytes:&num_heads       length:sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufNKV     = [ctx->device newBufferWithBytes:&num_kv_heads    length:sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufHD      = [ctx->device newBufferWithBytes:&head_dim        length:sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufPosOff  = [ctx->device newBufferWithBytes:&position_offset length:sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufSlide   = [ctx->device newBufferWithBytes:&sliding_window  length:sizeof(int32_t) options:MTLResourceStorageModeShared];

        if (!bufQ || !bufK || !bufV || !bufOut ||
            !bufSeqQ || !bufSeqKV || !bufNH || !bufNKV ||
            !bufHD || !bufPosOff || !bufSlide) return -7;

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        if (!cmd) return -8;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufQ      offset:0 atIndex:0];
        [enc setBuffer:bufK      offset:0 atIndex:1];
        [enc setBuffer:bufV      offset:0 atIndex:2];
        [enc setBuffer:bufOut    offset:0 atIndex:3];
        [enc setBuffer:bufSeqQ   offset:0 atIndex:4];
        [enc setBuffer:bufSeqKV  offset:0 atIndex:5];
        [enc setBuffer:bufNH     offset:0 atIndex:6];
        [enc setBuffer:bufNKV    offset:0 atIndex:7];
        [enc setBuffer:bufHD     offset:0 atIndex:8];
        [enc setBuffer:bufPosOff offset:0 atIndex:9];
        [enc setBuffer:bufSlide  offset:0 atIndex:10];

        // Dynamic threadgroup memory:
        //   q_shared(head_dim) + score_tile(256) + out_accum(head_dim) + warp_scratch(8)
        const int TILE_KV = 256;
        NSUInteger smemBytes = (NSUInteger)(2 * head_dim + TILE_KV + 8) * sizeof(float);
        [enc setThreadgroupMemoryLength:smemBytes atIndex:0];

        NSUInteger tgw     = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
        NSUInteger nGroups = (NSUInteger)(seq_q * num_heads);
        [enc dispatchThreadgroups:MTLSizeMake(nGroups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.error != nil) return -11;

        memcpy(output, bufOut.contents, outputBytes);
        return 0;
    }
}

extern "C" int dotllm_metal_attention_f16(
    dotllm_metal_context* ctx,
    const uint16_t* q,
    const uint16_t* k,
    const uint16_t* v,
    uint16_t*       output,
    int32_t         seq_q,
    int32_t         seq_kv,
    int32_t         num_heads,
    int32_t         num_kv_heads,
    int32_t         head_dim,
    int32_t         position_offset,
    int32_t         sliding_window)
{
    return run_attention_kernel(ctx, "attention_f16",
        q, k, v, output, sizeof(uint16_t),
        seq_q, seq_kv, num_heads, num_kv_heads, head_dim,
        position_offset, sliding_window);
}

extern "C" int dotllm_metal_attention_f32(
    dotllm_metal_context* ctx,
    const float* q,
    const float* k,
    const float* v,
    float*       output,
    int32_t      seq_q,
    int32_t      seq_kv,
    int32_t      num_heads,
    int32_t      num_kv_heads,
    int32_t      head_dim,
    int32_t      position_offset,
    int32_t      sliding_window)
{
    return run_attention_kernel(ctx, "attention_f32",
        q, k, v, output, sizeof(float),
        seq_q, seq_kv, num_heads, num_kv_heads, head_dim,
        position_offset, sliding_window);
}

// ── KV-cache quantization ────────────────────────────────────────────────────
//
// Both quant kernels share the same buffer layout:
//   buffer(0) = src  (FP16 input, read-only, passed as uint16_t*)
//   buffer(1) = dst  (quantized output bytes)
//   buffer(2) = total_blocks (constant int)
//
// Dispatch: one thread per block → dispatchThreads(total_blocks, 1, 1).
//
static int run_quant_kv_kernel(
    dotllm_metal_context* ctx,
    const char*      functionName,
    const uint16_t*  src,
    NSUInteger       src_bytes,
    uint8_t*         dst,
    NSUInteger       dst_bytes,
    int32_t          total_blocks)
{
    @autoreleasepool {
        if (!ctx || !src || !dst || total_blocks <= 0) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "quant_kv.metal", functionName);
        if (!pipeline) return -3;

        id<MTLBuffer> bufSrc   = [ctx->device newBufferWithBytes:src   length:src_bytes          options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDst   = [ctx->device newBufferWithLength:dst_bytes                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufTotal = [ctx->device newBufferWithBytes:&total_blocks length:sizeof(int32_t) options:MTLResourceStorageModeShared];
        if (!bufSrc || !bufDst || !bufTotal) return -7;

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        if (!cmd) return -8;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufSrc   offset:0 atIndex:0];
        [enc setBuffer:bufDst   offset:0 atIndex:1];
        [enc setBuffer:bufTotal offset:0 atIndex:2];

        // One thread per quantization block.
        NSUInteger tgw = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
        if ((NSUInteger)total_blocks < tgw) tgw = (NSUInteger)total_blocks;
        if (tgw == 0) tgw = 1;

        [enc dispatchThreads:MTLSizeMake(total_blocks, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.error != nil) return -11;

        memcpy(dst, bufDst.contents, dst_bytes);
        return 0;
    }
}

extern "C" int dotllm_metal_quant_f16_to_q8_0(
    dotllm_metal_context* ctx,
    const uint16_t* src,
    uint8_t*        dst,
    int32_t         total_blocks)
{
    const int32_t BLOCK_SIZE  = 32;
    const int32_t BLOCK_BYTES = 34;
    NSUInteger src_bytes = (NSUInteger)total_blocks * BLOCK_SIZE * sizeof(uint16_t);
    NSUInteger dst_bytes = (NSUInteger)total_blocks * BLOCK_BYTES;
    return run_quant_kv_kernel(ctx, "quant_f16_to_q8_0",
        src, src_bytes, dst, dst_bytes, total_blocks);
}

extern "C" int dotllm_metal_quant_f16_to_q4_0(
    dotllm_metal_context* ctx,
    const uint16_t* src,
    uint8_t*        dst,
    int32_t         total_blocks)
{
    const int32_t BLOCK_SIZE  = 32;
    const int32_t BLOCK_BYTES = 18;
    NSUInteger src_bytes = (NSUInteger)total_blocks * BLOCK_SIZE * sizeof(uint16_t);
    NSUInteger dst_bytes = (NSUInteger)total_blocks * BLOCK_BYTES;
    return run_quant_kv_kernel(ctx, "quant_f16_to_q4_0",
        src, src_bytes, dst, dst_bytes, total_blocks);
}

// ── Quantized GEMV ───────────────────────────────────────────────────────────

extern "C" int dotllm_metal_quantized_gemv_q8_0_f32in(
    dotllm_metal_context* ctx,
    const uint8_t* weight,
    const float*   x,
    float*         y,
    int32_t        n,
    int32_t        k)
{
    @autoreleasepool {
        if (!ctx || !weight || !x || !y || n <= 0 || k <= 0 || k % 32 != 0) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "quantized_gemv_f32in.metal",
                                       "quantized_gemv_q8_0_f32in");
        if (!pipeline) return -3;

        // weight: n rows × (k/32) blocks × 34 bytes
        int32_t bpr = k / 32;
        NSUInteger weightBytes = (NSUInteger)n * bpr * 34;
        NSUInteger xBytes      = (NSUInteger)k * sizeof(float);
        NSUInteger yBytes      = (NSUInteger)n * sizeof(float);

        id<MTLBuffer> bufW = [ctx->device newBufferWithBytes:weight length:weightBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufX = [ctx->device newBufferWithBytes:x      length:xBytes      options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufY = [ctx->device newBufferWithLength:yBytes                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufN = [ctx->device newBufferWithBytes:&n length:sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufK = [ctx->device newBufferWithBytes:&k length:sizeof(int32_t) options:MTLResourceStorageModeShared];
        if (!bufW || !bufX || !bufY || !bufN || !bufK) return -7;

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        if (!cmd) return -8;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufW offset:0 atIndex:0];
        [enc setBuffer:bufX offset:0 atIndex:1];
        [enc setBuffer:bufY offset:0 atIndex:2];
        [enc setBuffer:bufN offset:0 atIndex:3];
        [enc setBuffer:bufK offset:0 atIndex:4];

        // One threadgroup per output row; 256 threads per group.
        NSUInteger tgw = MIN(pipeline.maxTotalThreadsPerThreadgroup, 256);
        [enc dispatchThreadgroups:MTLSizeMake(n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.error != nil) return -11;

        memcpy(y, bufY.contents, yBytes);
        return 0;
    }
}

// ── Dequantization ───────────────────────────────────────────────────────────
//
// All dequant kernels share the same buffer layout:
//   buffer(0) = src  (raw quantized bytes, read-only)
//   buffer(1) = dst  (output half values)
//   buffer(2) = total_blocks or total_superblocks (constant int)
//
// Two dispatch strategies:
//   "warp-per-block"  (Q8_0 / Q4_0 / Q5_0): tgw=256, 8 warps/group → nGroups = ceil(total/8)
//   "thread-per-elem" (Q4_K / Q5_K / Q6_K): tgw=256,  1 group/superblock → nGroups = total
//
static int run_dequant_kernel(
    dotllm_metal_context* ctx,
    const char*    functionName,
    const uint8_t* src,
    NSUInteger     src_bytes,
    uint16_t*      dst,
    NSUInteger     dst_bytes,
    int32_t        total,       // total_blocks or total_superblocks
    NSUInteger     n_groups,
    NSUInteger     tgw)
{
    @autoreleasepool {
        if (!ctx || !src || !dst || total <= 0) return -10;

        id<MTLComputePipelineState> pipeline =
            get_or_create_pipeline(ctx, "dequant.metal", functionName);
        if (!pipeline) return -3;

        id<MTLBuffer> bufSrc   = [ctx->device newBufferWithBytes:src   length:src_bytes          options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufDst   = [ctx->device newBufferWithLength:dst_bytes                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufTotal = [ctx->device newBufferWithBytes:&total length:sizeof(int32_t)   options:MTLResourceStorageModeShared];
        if (!bufSrc || !bufDst || !bufTotal) return -7;

        id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
        if (!cmd) return -8;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!enc) return -9;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:bufSrc   offset:0 atIndex:0];
        [enc setBuffer:bufDst   offset:0 atIndex:1];
        [enc setBuffer:bufTotal offset:0 atIndex:2];

        [enc dispatchThreadgroups:MTLSizeMake(n_groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if (cmd.error != nil) return -11;

        memcpy(dst, bufDst.contents, dst_bytes);
        return 0;
    }
}

// threads-per-group for all dequant kernels
static const NSUInteger kDequantTgw = 256;
// warps-per-group for the warp-per-block kernels (256 threads / 32 per block)
static const NSUInteger kDequantWarpsPerGroup = 8;

extern "C" int dotllm_metal_dequant_q8_0_f16(
    dotllm_metal_context* ctx,
    const uint8_t* src,
    uint16_t*      dst,
    int32_t        total_blocks)
{
    const int32_t BLOCK_BYTES = 34, BLOCK_SIZE = 32;
    NSUInteger src_bytes = (NSUInteger)total_blocks * BLOCK_BYTES;
    NSUInteger dst_bytes = (NSUInteger)total_blocks * BLOCK_SIZE * sizeof(uint16_t);
    NSUInteger n_groups  = ((NSUInteger)total_blocks + kDequantWarpsPerGroup - 1) / kDequantWarpsPerGroup;
    return run_dequant_kernel(ctx, "dequant_q8_0_f16",
        src, src_bytes, dst, dst_bytes, total_blocks, n_groups, kDequantTgw);
}

extern "C" int dotllm_metal_dequant_q4_0_f16(
    dotllm_metal_context* ctx,
    const uint8_t* src,
    uint16_t*      dst,
    int32_t        total_blocks)
{
    const int32_t BLOCK_BYTES = 18, BLOCK_SIZE = 32;
    NSUInteger src_bytes = (NSUInteger)total_blocks * BLOCK_BYTES;
    NSUInteger dst_bytes = (NSUInteger)total_blocks * BLOCK_SIZE * sizeof(uint16_t);
    NSUInteger n_groups  = ((NSUInteger)total_blocks + kDequantWarpsPerGroup - 1) / kDequantWarpsPerGroup;
    return run_dequant_kernel(ctx, "dequant_q4_0_f16",
        src, src_bytes, dst, dst_bytes, total_blocks, n_groups, kDequantTgw);
}

extern "C" int dotllm_metal_dequant_q5_0_f16(
    dotllm_metal_context* ctx,
    const uint8_t* src,
    uint16_t*      dst,
    int32_t        total_blocks)
{
    const int32_t BLOCK_BYTES = 22, BLOCK_SIZE = 32;
    NSUInteger src_bytes = (NSUInteger)total_blocks * BLOCK_BYTES;
    NSUInteger dst_bytes = (NSUInteger)total_blocks * BLOCK_SIZE * sizeof(uint16_t);
    NSUInteger n_groups  = ((NSUInteger)total_blocks + kDequantWarpsPerGroup - 1) / kDequantWarpsPerGroup;
    return run_dequant_kernel(ctx, "dequant_q5_0_f16",
        src, src_bytes, dst, dst_bytes, total_blocks, n_groups, kDequantTgw);
}

extern "C" int dotllm_metal_dequant_q4_k_f16(
    dotllm_metal_context* ctx,
    const uint8_t* src,
    uint16_t*      dst,
    int32_t        total_superblocks)
{
    const int32_t BLOCK_BYTES = 144, SUPER_BLOCK_SIZE = 256;
    NSUInteger src_bytes = (NSUInteger)total_superblocks * BLOCK_BYTES;
    NSUInteger dst_bytes = (NSUInteger)total_superblocks * SUPER_BLOCK_SIZE * sizeof(uint16_t);
    NSUInteger n_groups  = (NSUInteger)total_superblocks; // one group per superblock
    return run_dequant_kernel(ctx, "dequant_q4_k_f16",
        src, src_bytes, dst, dst_bytes, total_superblocks, n_groups, kDequantTgw);
}

extern "C" int dotllm_metal_dequant_q5_k_f16(
    dotllm_metal_context* ctx,
    const uint8_t* src,
    uint16_t*      dst,
    int32_t        total_superblocks)
{
    const int32_t BLOCK_BYTES = 176, SUPER_BLOCK_SIZE = 256;
    NSUInteger src_bytes = (NSUInteger)total_superblocks * BLOCK_BYTES;
    NSUInteger dst_bytes = (NSUInteger)total_superblocks * SUPER_BLOCK_SIZE * sizeof(uint16_t);
    NSUInteger n_groups  = (NSUInteger)total_superblocks;
    return run_dequant_kernel(ctx, "dequant_q5_k_f16",
        src, src_bytes, dst, dst_bytes, total_superblocks, n_groups, kDequantTgw);
}

extern "C" int dotllm_metal_dequant_q6_k_f16(
    dotllm_metal_context* ctx,
    const uint8_t* src,
    uint16_t*      dst,
    int32_t        total_superblocks)
{
    const int32_t BLOCK_BYTES = 210, SUPER_BLOCK_SIZE = 256;
    NSUInteger src_bytes = (NSUInteger)total_superblocks * BLOCK_BYTES;
    NSUInteger dst_bytes = (NSUInteger)total_superblocks * SUPER_BLOCK_SIZE * sizeof(uint16_t);
    NSUInteger n_groups  = (NSUInteger)total_superblocks;
    return run_dequant_kernel(ctx, "dequant_q6_k_f16",
        src, src_bytes, dst, dst_bytes, total_superblocks, n_groups, kDequantTgw);
}

extern "C" int dotllm_metal_convert_f16_to_f32(
    dotllm_metal_context* ctx,
    const uint16_t* src,
    float*          dst,
    int32_t         n)
{
    return run_convert_kernel(ctx, "convert_f16_to_f32", src, dst, n, sizeof(uint16_t), sizeof(float));
}

extern "C" int dotllm_metal_convert_f32_to_f16(
    dotllm_metal_context* ctx,
    const float* src,
    uint16_t*    dst,
    int32_t      n)
{
    return run_convert_kernel(ctx, "convert_f32_to_f16", src, dst, n, sizeof(float), sizeof(uint16_t));
}
