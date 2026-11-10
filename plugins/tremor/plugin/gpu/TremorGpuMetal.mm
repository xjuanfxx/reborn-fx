// TremorGpuMetal.mm — Metal backend (macOS, Apple Silicon and Intel).
//
// The kernels are compiled from source when the device is set up. Built with ARC.
#include "TremorGpu.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <initializer_list>

#include "tremor_kernels_metal.h"  // generated: kTremorMetalSource

namespace tremor::gpu {
namespace {

class MetalBackend final : public Backend {
public:
    MetalBackend(id<MTLDevice> device, id<MTLCommandQueue> queue) : device_(device), queue_(queue) {}

    bool init() {
        @autoreleasepool {
            NSError* error = nil;
            NSString* source = [NSString stringWithUTF8String:kTremorMetalSource];
            id<MTLLibrary> library = [device_ newLibraryWithSource:source options:nil error:&error];
            if (!library) return false;
            source_ = pipeline(library, @"tremor_source");
            rotate_ = pipeline(library, @"tremor_rotate");
            finish_ = pipeline(library, @"tremor_finish");
            return source_ && rotate_ && finish_;
        }
    }

    bool source(void* src, void* dst, const TremorSourceParams& p) override {
        return run(source_, {src, dst}, &p, sizeof(p), p.w, p.h);
    }
    bool rotate(void* src, void* dst, const TremorRotateParams& p) override {
        return run(rotate_, {src, dst}, &p, sizeof(p), p.w, p.h);
    }
    bool finish(void* rotated, void* source, void* dst, const TremorFinishParams& p) override {
        return run(finish_, {rotated, source, dst}, &p, sizeof(p), p.out_w, p.out_h);
    }

private:
    id<MTLComputePipelineState> pipeline(id<MTLLibrary> library, NSString* name) {
        id<MTLFunction> fn = [library newFunctionWithName:name];
        if (!fn) return nil;
        NSError* error = nil;
        return [device_ newComputePipelineStateWithFunction:fn error:&error];
    }

    bool run(id<MTLComputePipelineState> state, std::initializer_list<void*> buffers, const void* params,
             size_t params_size, int w, int h) {
        @autoreleasepool {
            id<MTLCommandBuffer> commands = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
            [encoder setComputePipelineState:state];
            NSUInteger index = 0;
            for (void* buffer : buffers) {
                [encoder setBuffer:(__bridge id<MTLBuffer>)buffer offset:0 atIndex:index++];
            }
            [encoder setBytes:params length:params_size atIndex:index];
            const NSUInteger tw = state.threadExecutionWidth;
            const NSUInteger th = std::max<NSUInteger>(1, state.maxTotalThreadsPerThreadgroup / tw);
            const MTLSize group = MTLSizeMake(tw, th, 1);
            const MTLSize groups = MTLSizeMake((NSUInteger(w) + tw - 1) / tw, (NSUInteger(h) + th - 1) / th, 1);
            [encoder dispatchThreadgroups:groups threadsPerThreadgroup:group];
            [encoder endEncoding];
            [commands commit];
            return commands.error == nil;
        }
    }

    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLComputePipelineState> source_ = nil, rotate_ = nil, finish_ = nil;
};

}  // namespace

std::unique_ptr<Backend> make_metal(void* device, void* queue) {
    if (!device || !queue) return nullptr;
    auto backend = std::make_unique<MetalBackend>((__bridge id<MTLDevice>)device, (__bridge id<MTLCommandQueue>)queue);
    if (!backend->init()) return nullptr;
    return backend;
}

}  // namespace tremor::gpu
