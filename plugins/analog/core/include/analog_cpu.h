// analog_cpu.h — CPU renderer: runs analog_kernels.inc as C++.
//
// The GPU backends run the very same source, so both paths agree up to float
// rounding (tools/gpu_check measures it).
//
// Buffers are float4 in After Effects' GPU layout (x = B, y = G, z = R, w = A),
// premultiplied, 0–1 at 8/16 bpc. The plugin converts AE worlds to and from it.
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "analog_core.h"

namespace analog {
namespace kernels {

struct float4 {
    float x, y, z, w;
};

#define DEVICE static inline
#define GBUF float4*
#define GBUF_CONST const float4*
#define F4(a, b, c, d) float4{(a), (b), (c), (d)}
#define UINT uint32_t
#define AN_PARAMS const AnalogParams&
#define FSIN std::sin
#define FCOS std::cos
#define FFLOOR std::floor
#define FABS std::fabs
#define KERNEL1(name, P) inline void name(GBUF_CONST src, GBUF dst, const P& p, int x, int y) {
#define KERNEL2(name, P) inline void name(GBUF_CONST src, GBUF_CONST src2, GBUF dst, const P& p, int x, int y) {
#define KERNEL_END }

#include "analog_kernels.inc"

#undef DEVICE
#undef GBUF
#undef GBUF_CONST
#undef F4
#undef UINT
#undef AN_PARAMS
#undef FSIN
#undef FCOS
#undef FFLOOR
#undef FABS
#undef KERNEL1
#undef KERNEL2
#undef KERNEL_END

}  // namespace kernels

using kernels::float4;

// Renders one frame. `in` is in_w x in_h (pitch p.in_pitch), `out` is out_w x
// out_h (pitch p.out_pitch).
inline void render_cpu(const AnalogParams& params, const float4* in, float4* out) {
    AnalogParams p = params;
    p.plane_sx = 1;
    p.plane_sy = p.line_w;
    const size_t plane = size_t(p.line_w) * size_t(p.rows);
    std::vector<float4> a(plane), b(plane);

    parallel_for(p.rows, [&](int y) {
        for (int x = 0; x < p.line_w; ++x) kernels::analog_prepare(in, a.data(), p, x, y);
    });
    parallel_for(p.rows, [&](int row) { kernels::analog_line_a(a.data(), a.data(), p, row, 0); });
    parallel_for(p.rows, [&](int row) { kernels::analog_line_b(a.data(), b.data(), p, row, 0); });

    if (p.blur > 1.0f) {
        std::vector<float4> tmp(size_t(p.canvas_w) * size_t(p.rows));
        std::vector<float4> lowres(size_t(p.blur_w) * size_t(p.blur_h));
        AnalogParams pt = p;
        pt.out_w = p.canvas_w;
        pt.out_h = p.rows;
        pt.out_ox = pt.out_oy = 0;
        pt.out_pitch = p.canvas_w;
        parallel_for(p.rows, [&](int y) {
            for (int x = 0; x < p.canvas_w; ++x) kernels::analog_output(b.data(), in, tmp.data(), pt, x, y);
        });
        p.tmp_pitch = p.canvas_w;
        p.small_pitch = p.blur_w;
        parallel_for(p.blur_h, [&](int y) {
            for (int x = 0; x < p.blur_w; ++x) kernels::analog_blur_down(tmp.data(), lowres.data(), p, x, y);
        });
        parallel_for(p.out_h, [&](int y) {
            for (int x = 0; x < p.out_w; ++x) kernels::analog_blur_up(lowres.data(), out, p, x, y);
        });
    } else {
        parallel_for(p.out_h, [&](int y) {
            for (int x = 0; x < p.out_w; ++x) kernels::analog_output(b.data(), in, out, p, x, y);
        });
    }
}

}  // namespace analog
