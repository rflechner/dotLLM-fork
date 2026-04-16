#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <string.h>

static int run_binary_f32_kernel(
    const char* shaderPath,
    const char* functionName,
    const float* a,
    const float* b,
    float* result,
    uint32_t length)
{
    @autoreleasepool {
        if (a == nullptr || b == nullptr || result == nullptr) return -10;

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return -1;

        NSError* error = nil;
        NSString* source = [NSString stringWithContentsOfFile:@(shaderPath)
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
        if (!source) return -2;

        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) return -3;

        id<MTLFunction> function = [library newFunctionWithName:@(functionName)];
        if (!function) return -4;

        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) return -5;

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) return -6;

        NSUInteger bytes = (NSUInteger)length * sizeof(float);
        id<MTLBuffer> bufA = [device newBufferWithBytes:a length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB = [device newBufferWithBytes:b length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufR = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufL = [device newBufferWithBytes:&length length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        if (!bufA || !bufB || !bufR || !bufL) return -7;

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
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

extern "C" int dotllm_metal_add_f32(
    const float* a,
    const float* b,
    float* result,
    uint32_t length)
{
    return run_binary_f32_kernel("add.metal", "add_arrays", a, b, result, length);
}

extern "C" int dotllm_metal_multiply_f32(
    const float* a,
    const float* b,
    float* result,
    uint32_t length)
{
    return run_binary_f32_kernel("multiply.metal", "multiply_arrays", a, b, result, length);
}
