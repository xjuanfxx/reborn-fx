// AnalogGpuRender.h — runs the passes of analog_kernels.inc on the GPU.
//
// Independent of After Effects: the plugin and tools/gpu_check both use it.
// Mirrors analog::render_cpu (core/include/analog_cpu.h).
#pragma once

#include "AnalogGpu.h"
#include "analog_types.h"

namespace analog::gpu {

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

inline bool render(Backend& gpu, Scratch& scratch, const AnalogParams& params, const Buffer& in, const Buffer& out) {
    AnalogParams p = params;
    p.in_pitch = in.pitch;
    p.out_pitch = out.pitch;

    // Line buffers are stored transposed (one column per row) so that the
    // per-row kernels, one thread per row, read neighbouring rows together.
    Buffer a, b;
    if (!scratch.alloc(p.rows, p.line_w, a) || !scratch.alloc(p.rows, p.line_w, b)) return false;
    if (a.pitch != b.pitch) return false;
    p.plane_sx = a.pitch;
    p.plane_sy = 1;

    if (!gpu.run(K_PREPARE, in.mem, a.mem, nullptr, p, p.line_w, p.rows)) return false;
    if (!gpu.run(K_LINE_A, a.mem, a.mem, nullptr, p, p.rows, 1)) return false;
    if (!gpu.run(K_LINE_B, a.mem, b.mem, nullptr, p, p.rows, 1)) return false;

    if (p.blur > 1.0f) {
        Buffer tmp, lowres;
        if (!scratch.alloc(p.canvas_w, p.rows, tmp) || !scratch.alloc(p.blur_w, p.blur_h, lowres)) return false;
        AnalogParams pt = p;
        pt.out_w = p.canvas_w;
        pt.out_h = p.rows;
        pt.out_ox = pt.out_oy = 0;
        pt.out_pitch = tmp.pitch;
        if (!gpu.run(K_OUTPUT, b.mem, in.mem, tmp.mem, pt, p.canvas_w, p.rows)) return false;
        p.tmp_pitch = tmp.pitch;
        p.small_pitch = lowres.pitch;
        if (!gpu.run(K_BLUR_DOWN, tmp.mem, lowres.mem, nullptr, p, p.blur_w, p.blur_h)) return false;
        return gpu.run(K_BLUR_UP, lowres.mem, out.mem, nullptr, p, p.out_w, p.out_h);
    }
    return gpu.run(K_OUTPUT, b.mem, in.mem, out.mem, p, p.out_w, p.out_h);
}

}  // namespace analog::gpu
