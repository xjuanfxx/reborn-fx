// TremorGpu.h — GPU backends behind one interface.
//
// After Effects picks the GPU framework (CUDA or OpenCL on Windows, Metal on
// macOS); each backend runs the same kernels (tremor_kernels.inc). Buffers are
// the raw GPU memory handles After Effects hands out for its GPU worlds
// (CUdeviceptr, cl_mem or id<MTLBuffer>).
#pragma once

#include <memory>

#include "tremor_gpu_types.h"

namespace tremor::gpu {

class Backend {
public:
    virtual ~Backend() = default;
    // Each call enqueues one pass on the host's queue; false on failure.
    virtual bool source(void* src, void* dst, const TremorSourceParams& p) = 0;
    virtual bool rotate(void* src, void* dst, const TremorRotateParams& p) = 0;
    virtual bool finish(void* rotated, void* source, void* dst, const TremorFinishParams& p) = 0;
};

// Each factory returns nullptr if the framework or the kernels are unavailable;
// After Effects then renders Tremor on the CPU.
std::unique_ptr<Backend> make_cuda(void* context, void* stream);
std::unique_ptr<Backend> make_opencl(void* context, void* device, void* queue);
std::unique_ptr<Backend> make_metal(void* device, void* queue);

}  // namespace tremor::gpu
