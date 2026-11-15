// AnalogGpu.h — GPU backends behind one interface.
//
// After Effects picks the GPU framework (CUDA or OpenCL on Windows, Metal on
// macOS); each backend runs the same kernels (core/include/analog_kernels.inc).
// Buffers are the raw GPU memory handles After Effects hands out for its GPU
// worlds (CUdeviceptr, cl_mem or id<MTLBuffer>).
#pragma once

#include <memory>

#include "analog_types.h"

namespace analog::gpu {

enum Kernel { K_PREPARE, K_LINE_A, K_LINE_B, K_OUTPUT, K_BLUR_DOWN, K_BLUR_UP, K_COUNT };

inline const char* kernel_name(Kernel k) {
    static const char* const names[K_COUNT] = {"analog_prepare",   "analog_line_a",    "analog_line_b",
                                               "analog_output",    "analog_blur_down", "analog_blur_up"};
    return names[k];
}

// K_OUTPUT takes three buffers (src, src2, dst); the others two (src, dst).
inline bool three_buffers(Kernel k) { return k == K_OUTPUT; }

class Backend {
public:
    virtual ~Backend() = default;
    // Enqueues one kernel over a w x h grid on the host's queue; false on failure.
    // h == 1 marks the per-row kernels (one thread per row).
    virtual bool run(Kernel k, void* a, void* b, void* c, const AnalogParams& p, int w, int h) = 0;
};

// Each factory returns nullptr if the framework or the kernels are unavailable;
// After Effects then renders Analog on the CPU.
std::unique_ptr<Backend> make_cuda(void* context, void* stream);
std::unique_ptr<Backend> make_opencl(void* context, void* device, void* queue);
std::unique_ptr<Backend> make_metal(void* device, void* queue);

}  // namespace analog::gpu
