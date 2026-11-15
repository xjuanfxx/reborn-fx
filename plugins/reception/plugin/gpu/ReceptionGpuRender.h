// ReceptionGpuRender.h — runs the passes of reception_kernels.inc on the GPU.
//
// Independent of After Effects: the plugin and tools/gpu_check both use it.
// Mirrors reception::render_cpu (core/include/reception_cpu.h).
#pragma once

#include "ReceptionGpu.h"
#include "reception_types.h"

namespace reception::gpu {

// A float4 GPU buffer; pitch in pixels.
struct Buffer {
    void* mem = nullptr;
    int width = 0, height = 0, pitch = 0;
};

// Provides temporary buffers for one render; frees them when destroyed.
class Scratch {
public:
    virtual ~Scratch() = default;
    virtual bool alloc(int width, int height, Buffer& out) = 0;
};

inline bool render(Backend& gpu, Scratch& scratch, const ReceptionParams& params, const Buffer& in, const Buffer& out) {
    ReceptionParams p = params;
    p.in_pitch = in.pitch;
    p.out_pitch = out.pitch;
    const int w = p.canvas_w, h = p.canvas_h;

    Buffer table, d;
    if (!scratch.alloc(p.full_h, 1, table) || !scratch.alloc(w, h, d)) return false;
    p.d_pitch = d.pitch;
    if (!gpu.run(K_ROWS, in.mem, in.mem, in.mem, table.mem, p, 1, 1)) return false;
    if (!gpu.run(K_DISTORT, in.mem, table.mem, in.mem, d.mem, p, w, h)) return false;

    Buffer glow = d;
    if (p.glow_on) {
        Buffer t;
        if (!scratch.alloc(w, h, t) || !scratch.alloc(w, h, glow)) return false;
        p.t_pitch = t.pitch;
        p.g_pitch = glow.pitch;
        if (!gpu.run(K_GLOW_H, d.mem, d.mem, d.mem, t.mem, p, w, h)) return false;
        if (!gpu.run(K_GLOW_V, t.mem, t.mem, t.mem, glow.mem, p, w, h)) return false;
    } else {
        p.g_pitch = d.pitch;
    }
    return gpu.run(K_FINAL, d.mem, glow.mem, in.mem, out.mem, p, p.out_w, p.out_h);
}

}  // namespace reception::gpu
