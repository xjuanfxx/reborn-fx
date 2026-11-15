// ReceptionGpu.h — GPU backends behind one interface.
//
// After Effects picks the GPU framework (CUDA or OpenCL on Windows, Metal on
// macOS); each backend runs the same kernels (core/include/reception_kernels.inc).
// Buffers are the raw GPU memory handles After Effects hands out for its GPU
// worlds (CUdeviceptr, cl_mem or id<MTLBuffer>).
#pragma once

#include <memory>

#include "reception_types.h"

namespace reception::gpu {

enum Kernel { K_ROWS, K_DISTORT, K_GLOW_H, K_GLOW_V, K_FINAL, K_COUNT };

inline const char* kernel_name(Kernel k) {
    static const char* const names[K_COUNT] = {"reception_rows", "reception_distort", "reception_glow_h",
                                               "reception_glow_v", "reception_final"};
    return names[k];
}

class Backend {
public:
    virtual ~Backend() = default;
    // Enqueues one kernel over a w x h grid on the host's queue; false on failure.
    // Every kernel takes three input buffers and one output buffer.
    virtual bool run(Kernel k, void* a, void* b, void* c, void* dst, const ReceptionParams& p, int w, int h) = 0;
};

// Each factory returns nullptr if the framework or the kernels are unavailable;
// After Effects then renders Reception on the CPU.
std::unique_ptr<Backend> make_cuda(void* context, void* stream);
std::unique_ptr<Backend> make_opencl(void* context, void* device, void* queue);
std::unique_ptr<Backend> make_metal(void* device, void* queue);

}  // namespace reception::gpu
