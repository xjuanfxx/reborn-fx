// CUDA entry point for reception_kernels.inc. Compiled to PTX at build time and
// loaded through the CUDA driver API, so the plugin has no CUDA runtime dependency.
#define DEVICE __device__ __forceinline__
#define GBUF_CONST const float4*
#define GBUF float4*
#define F4(a, b, c, d) make_float4((a), (b), (c), (d))
#define UINT unsigned int
#define RC_PARAMS const ReceptionParams&
#define FSIN sinf
#define FPOW powf
#define FFLOOR floorf
#define FMIN fminf
#define FMAX fmaxf
#define KERNEL(name, P) \
    extern "C" __global__ void name(GBUF_CONST a, GBUF_CONST b, GBUF_CONST c, GBUF dst, P p) { \
        const int x = int(blockIdx.x * blockDim.x + threadIdx.x); \
        const int y = int(blockIdx.y * blockDim.y + threadIdx.y);
#define KERNEL_END }

#include "reception_types.h"
#include "reception_kernels.inc"
