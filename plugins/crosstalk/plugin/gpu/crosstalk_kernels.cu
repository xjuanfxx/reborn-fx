// CUDA entry point for crosstalk_kernels.inc. Compiled to PTX at build time and
// loaded through the CUDA driver API, so the plugin has no CUDA runtime dependency.
#define DEVICE __device__ __forceinline__
#define GBUF_CONST const float4*
#define GBUF float4*
#define F4(a, b, c, d) make_float4((a), (b), (c), (d))
#define UINT unsigned int
#define CT_PARAMS const CrosstalkParams&
#define FSIN sinf
#define FFLOOR floorf
#define FMIN fminf
#define FMAX fmaxf
#define KERNEL(name, P) \
    extern "C" __global__ void name(GBUF_CONST a, GBUF_CONST b, GBUF_CONST c, GBUF dst, P p) { \
        const int x = int(blockIdx.x * blockDim.x + threadIdx.x); \
        const int y = int(blockIdx.y * blockDim.y + threadIdx.y);
#define KERNEL_END }

#include "crosstalk_types.h"
#include "crosstalk_kernels.inc"
