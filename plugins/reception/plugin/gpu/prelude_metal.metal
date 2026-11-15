// Metal Shading Language prelude for reception_kernels.inc.
#include <metal_stdlib>
using namespace metal;

#define DEVICE static inline
#define GBUF_CONST device const float4*
#define GBUF device float4*
#define F4(a, b, c, d) float4((a), (b), (c), (d))
#define UINT uint
#define RC_PARAMS thread const ReceptionParams&
#define FSIN precise::sin
#define FPOW precise::pow
#define FFLOOR floor
#define FMIN fmin
#define FMAX fmax
#define KERNEL(name, P) \
    kernel void name(GBUF_CONST a [[buffer(0)]], GBUF_CONST b [[buffer(1)]], GBUF_CONST c [[buffer(2)]], \
                     GBUF dst [[buffer(3)]], constant P& p_in [[buffer(4)]], \
                     uint2 gid [[thread_position_in_grid]]) { \
        const P p = p_in; \
        const int x = int(gid.x); \
        const int y = int(gid.y);
#define KERNEL_END }
