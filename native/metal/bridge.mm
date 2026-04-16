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

