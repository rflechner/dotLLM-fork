#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <string.h>

extern "C" int dotllm_metal_add_f32(
    const float* a,
    const float* b,
    float* result,
    uint32_t length)
{
    @autoreleasepool {
        if (a == nullptr || b == nullptr || result == nullptr) {
            return -10;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            return -1;
        }

        NSError* error = nil;

        NSString* shaderPath = @"add.metal";
        NSString* source = [NSString stringWithContentsOfFile:shaderPath
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];

        if (!source) {
            return -2;
        }

        id<MTLLibrary> library = [device newLibraryWithSource:source
                                                      options:nil
                                                        error:&error];
        if (!library) {
            return -3;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"add_arrays"];
        if (!function) {
            return -4;
        }

        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            return -5;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            return -6;
        }

        NSUInteger bytes = (NSUInteger)length * sizeof(float);

        id<MTLBuffer> bufferA = [device newBufferWithBytes:a
                                                   length:bytes
                                                  options:MTLResourceStorageModeShared];

        id<MTLBuffer> bufferB = [device newBufferWithBytes:b
                                                   length:bytes
                                                  options:MTLResourceStorageModeShared];

        id<MTLBuffer> bufferR = [device newBufferWithLength:bytes
                                                    options:MTLResourceStorageModeShared];

        uint32_t lenValue = length;
        id<MTLBuffer> bufferLength = [device newBufferWithBytes:&lenValue
                                                         length:sizeof(uint32_t)
                                                        options:MTLResourceStorageModeShared];

        if (!bufferA || !bufferB || !bufferR || !bufferLength) {
            return -7;
        }

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        if (!commandBuffer) {
            return -8;
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!encoder) {
            return -9;
        }

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:bufferA offset:0 atIndex:0];
        [encoder setBuffer:bufferB offset:0 atIndex:1];
        [encoder setBuffer:bufferR offset:0 atIndex:2];
        [encoder setBuffer:bufferLength offset:0 atIndex:3];

        MTLSize gridSize = MTLSizeMake(length, 1, 1);

        NSUInteger threadgroupWidth = pipeline.maxTotalThreadsPerThreadgroup;
        if (threadgroupWidth > 256) {
            threadgroupWidth = 256;
        }
        if (threadgroupWidth > length && length > 0) {
            threadgroupWidth = length;
        }
        if (threadgroupWidth == 0) {
            threadgroupWidth = 1;
        }

        MTLSize threadgroupSize = MTLSizeMake(threadgroupWidth, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        if (commandBuffer.error != nil) {
            return -11;
        }

        memcpy(result, bufferR.contents, bytes);
        return 0;
    }
}
