// reception_cpu.h — CPU renderer: runs reception_kernels.inc as C++.
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

#include "reception_core.h"

namespace reception {
namespace kernels {

struct float4 {
    float x, y, z, w;
};

#define DEVICE static inline
#define GBUF float4*
#define GBUF_CONST const float4*
#define F4(a, b, c, d) float4{(a), (b), (c), (d)}
#define UINT uint32_t
#define RC_PARAMS const ReceptionParams&
#define FSIN std::sin
#define FPOW std::pow
#define FFLOOR std::floor
#define FMIN std::fmin
#define FMAX std::fmax
#define KERNEL(name, P) \
    inline void name(GBUF_CONST a, GBUF_CONST b, GBUF_CONST c, GBUF dst, const P& p, int x, int y) { \
        (void)a; (void)b; (void)c;
#define KERNEL_END }

#include "reception_kernels.inc"

#undef DEVICE
#undef GBUF
#undef GBUF_CONST
#undef F4
#undef UINT
#undef RC_PARAMS
#undef FSIN
#undef FPOW
#undef FFLOOR
#undef FMIN
#undef FMAX
#undef KERNEL
#undef KERNEL_END

}  // namespace kernels

using kernels::float4;

// Renders one frame. `in` is in_w x in_h (pitch p.in_pitch), `out` is out_w x
// out_h (pitch p.out_pitch).
inline void render_cpu(const ReceptionParams& params, const float4* in, float4* out) {
    ReceptionParams p = params;
    const int w = p.canvas_w, h = p.canvas_h;
    p.d_pitch = p.t_pitch = p.g_pitch = w;
    std::vector<float4> table(size_t(p.full_h)), d(size_t(w) * h);
    kernels::reception_rows(in, in, in, table.data(), p, 0, 0);
    parallel_for(h, [&](int y) {
        for (int x = 0; x < w; ++x) kernels::reception_distort(in, table.data(), in, d.data(), p, x, y);
    });
    std::vector<float4> glow;
    if (p.glow_on) {
        std::vector<float4> t(size_t(w) * h);
        glow.resize(size_t(w) * h);
        parallel_for(h, [&](int y) {
            for (int x = 0; x < w; ++x) kernels::reception_glow_h(d.data(), in, in, t.data(), p, x, y);
        });
        parallel_for(h, [&](int y) {
            for (int x = 0; x < w; ++x) kernels::reception_glow_v(t.data(), in, in, glow.data(), p, x, y);
        });
    }
    const float4* g = p.glow_on ? glow.data() : d.data();
    parallel_for(p.out_h, [&](int y) {
        for (int x = 0; x < p.out_w; ++x) kernels::reception_final(d.data(), g, in, out, p, x, y);
    });
}

}  // namespace reception
