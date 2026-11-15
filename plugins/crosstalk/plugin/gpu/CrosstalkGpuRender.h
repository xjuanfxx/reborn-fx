// CrosstalkGpuRender.h — runs crosstalk_kernels.inc on the GPU.
//
// Independent of After Effects: the plugin and tools/gpu_check both use it.
// Mirrors crosstalk::render_cpu (core/include/crosstalk_cpu.h).
#pragma once

#include "CrosstalkGpu.h"
#include "crosstalk_types.h"

namespace crosstalk::gpu {

// A float4 GPU buffer; pitch in pixels.
struct Buffer {
    void* mem = nullptr;
    int width = 0, height = 0, pitch = 0;
};

inline bool render(Backend& gpu, const CrosstalkParams& params, const Buffer& in, const Buffer& out) {
    CrosstalkParams p = params;
    p.in_pitch = in.pitch;
    p.out_pitch = out.pitch;
    return gpu.run(K_RENDER, in.mem, in.mem, in.mem, out.mem, p, p.out_w, p.out_h);
}

}  // namespace crosstalk::gpu
