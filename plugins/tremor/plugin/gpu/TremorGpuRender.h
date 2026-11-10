// TremorGpuRender.h — turns a FrameState into GPU passes.
//
// Independent of After Effects: the plugin and tools/gpu_check both use it, so
// the GPU path can be compared against the CPU renderer outside AE.
// Mirrors tremor::render_frame (core/include/tremor_render.h).
#pragma once

#include <cfloat>
#include <cmath>

#include "TremorGpu.h"
#include "tremor_render.h"

namespace tremor::gpu {

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

// Where the input and output buffers sit on the whole-layer canvas.
struct Layout {
    int canvas_w = 0, canvas_h = 0;
    int in_ox = 0, in_oy = 0;   // input buffer corner, in canvas coords
    int out_ox = 0, out_oy = 0; // output pixel (0,0), in canvas coords
};

inline bool render(Backend& gpu, Scratch& scratch, const FrameState& st, const Look& look, const Layout& l,
                   const Buffer& in, const Buffer& out) {
    Buffer source, rotated;
    if (!scratch.alloc(l.canvas_w, l.canvas_h, source)) return false;

    TremorSourceParams sp{};
    sp.w = l.canvas_w;
    sp.h = l.canvas_h;
    sp.dst_pitch = source.pitch;
    sp.in_w = in.width;
    sp.in_h = in.height;
    sp.in_pitch = in.pitch;
    sp.in_ox = l.in_ox;
    sp.in_oy = l.in_oy;
    sp.mode = int(look.deinterlace);
    if (!gpu.source(in.mem, source.mem, sp)) return false;

    const bool copy_source = look.blend >= 1.0f;
    if (copy_source) {
        rotated = source;  // not read
    } else {
        if (!scratch.alloc(l.canvas_w, l.canvas_h, rotated)) return false;
        TremorRotateParams rp{};
        rp.w = l.canvas_w;
        rp.h = l.canvas_h;
        rp.src_pitch = source.pitch;
        rp.dst_pitch = rotated.pitch;
        rp.edges = int(look.edges);
        rp.cx = float(st.center_x);
        rp.cy = float(st.center_y);
        const Turns t = turns(st);
        rp.c0 = float(t.c[0]);
        rp.s0 = float(t.s[0]);
        rp.c1 = float(t.c[1]);
        rp.s1 = float(t.s[1]);
        rp.c2 = float(t.c[2]);
        rp.s2 = float(t.s[2]);
        if (!gpu.rotate(source.mem, rotated.mem, rp)) return false;
    }

    TremorFinishParams fp{};
    fp.out_w = out.width;
    fp.out_h = out.height;
    fp.out_pitch = out.pitch;
    fp.w = l.canvas_w;
    fp.h = l.canvas_h;
    fp.rot_pitch = rotated.pitch;
    fp.src_pitch = source.pitch;
    fp.ox = l.out_ox;
    fp.oy = l.out_oy;
    fp.edges = int(look.edges);
    fp.composite = look.alpha_composite ? 1 : 0;
    const double inv = 1.0 / (st.zoom + 1.0);
    fp.zoom_valid = std::isfinite(inv) ? 1 : 0;
    fp.inv_zoom = fp.zoom_valid ? float(inv) : 0.0f;
    fp.cx = float(st.center_x);
    fp.cy = float(st.center_y);
    fp.shift_x = float(st.shift_x);
    fp.shift_y = float(st.shift_y);
    fp.adj_r = look.adjust[0];
    fp.adj_g = look.adjust[1];
    fp.adj_b = look.adjust[2];
    fp.blend = look.blend;
    fp.color_max = std::isfinite(look.color_max) ? look.color_max : FLT_MAX;
    return gpu.finish(rotated.mem, source.mem, out.mem, fp);
}

}  // namespace tremor::gpu
