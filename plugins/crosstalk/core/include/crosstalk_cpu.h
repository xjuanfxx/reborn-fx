// crosstalk_cpu.h — CPU renderer: runs crosstalk_kernels.inc as C++.
//
// The GPU backends run the very same source, so both paths agree up to float
// rounding (tools/gpu_check measures it).
//
// Buffers are float4 in After Effects' GPU layout (x = B, y = G, z = R, w = A),
// premultiplied, 0–1 at 8/16 bpc. The plugin converts AE worlds to and from it.
#pragma once

#include <cmath>
#include <cstdint>

#include "crosstalk_core.h"

namespace crosstalk {
namespace kernels {

struct float4 {
    float x, y, z, w;
};

#define DEVICE static inline
#define GBUF float4*
#define GBUF_CONST const float4*
#define F4(a, b, c, d) float4{(a), (b), (c), (d)}
#define UINT uint32_t
#define CT_PARAMS const CrosstalkParams&
#define FSIN std::sin
#define FFLOOR std::floor
#define FMIN std::fmin
#define FMAX std::fmax
#define KERNEL(name, P) \
    inline void name(GBUF_CONST a, GBUF_CONST b, GBUF_CONST c, GBUF dst, const P& p, int x, int y) { \
        (void)b; (void)c;
#define KERNEL_END }

#include "crosstalk_kernels.inc"

#undef DEVICE
#undef GBUF
#undef GBUF_CONST
#undef F4
#undef UINT
#undef CT_PARAMS
#undef FSIN
#undef FFLOOR
#undef FMIN
#undef FMAX
#undef KERNEL
#undef KERNEL_END

}  // namespace kernels

using kernels::float4;

// Renders one frame. `in` is in_w x in_h (pitch p.in_pitch), `out` is out_w x
// out_h (pitch p.out_pitch).
inline void render_cpu(const CrosstalkParams& p, const float4* in, float4* out) {
    parallel_for(p.out_h, [&](int y) {
        for (int x = 0; x < p.out_w; ++x) kernels::crosstalk_render(in, in, in, out, p, x, y);
    });
}

}  // namespace crosstalk
