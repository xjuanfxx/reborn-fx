// AnalogGpuMetal.mm — Metal backend (macOS, Apple Silicon and Intel).
//
// The kernels are compiled from source when the device is set up. Built with ARC.
#include "AnalogGpu.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>

#include "analog_kernels_metal.h"  // generated: kAnalogMetalSource

namespace analog::gpu {
namespace {

class MetalBackend final : public Backend {
public:
    MetalBackend(id<MTLDevice> device, id<MTLCommandQueue> queue) : device_(device), queue_(queue) {}

    bool init() {
        @autoreleasepool {
            NSError* error = nil;
            MTLCompileOptions* options = [MTLCompileOptions new];
            options.fastMathEnabled = NO;  // sin() must stay accurate for the FM luma
            NSString* source = [NSString stringWithUTF8String:kAnalogMetalSource];
            id<MTLLibrary> library = [device_ newLibraryWithSource:source options:options error:&error];
            if (!library) return false;
            for (int k = 0; k < K_COUNT; ++k) {
                id<MTLFunction> fn = [library newFunctionWithName:@(kernel_name(Kernel(k)))];
                if (!fn) return false;
                pipelines_[k] = [device_ newComputePipelineStateWithFunction:fn error:&error];
                if (!pipelines_[k]) return false;
            }
            return true;
        }
    }

    bool run(Kernel k, void* a, void* b, void* c, const AnalogParams& p, int w, int h) override {
        if (w <= 0 || h <= 0) return true;
        @autoreleasepool {
            id<MTLComputePipelineState> state = pipelines_[k];
            id<MTLCommandBuffer> commands = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
            [encoder setComputePipelineState:state];
            NSUInteger index = 0;
            [encoder setBuffer:(__bridge id<MTLBuffer>)a offset:0 atIndex:index++];
            [encoder setBuffer:(__bridge id<MTLBuffer>)b offset:0 atIndex:index++];
            if (three_buffers(k)) [encoder setBuffer:(__bridge id<MTLBuffer>)c offset:0 atIndex:index++];
            [encoder setBytes:&p length:sizeof(p) atIndex:index];
            // Per-row kernels (h == 1): one row of threads; image kernels: 2-D groups.
            const NSUInteger tw = state.threadExecutionWidth;
            const NSUInteger th = h == 1 ? 1 : std::max<NSUInteger>(1, state.maxTotalThreadsPerThreadgroup / tw);
            const MTLSize group = MTLSizeMake(tw, th, 1);
            const MTLSize groups = MTLSizeMake((NSUInteger(w) + tw - 1) / tw, (NSUInteger(h) + th - 1) / th, 1);
            [encoder dispatchThreadgroups:groups threadsPerThreadgroup:group];
            [encoder endEncoding];
            [commands commit];
            return commands.error == nil;
        }
    }

private:
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLComputePipelineState> pipelines_[K_COUNT] = {};
};

}  // namespace

std::unique_ptr<Backend> make_metal(void* device, void* queue) {
    if (!device || !queue) return nullptr;
    auto backend = std::make_unique<MetalBackend>((__bridge id<MTLDevice>)device, (__bridge id<MTLCommandQueue>)queue);
    if (!backend->init()) return nullptr;
    return backend;
}

}  // namespace analog::gpu
