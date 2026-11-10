// CUDA entry point for tremor_kernels.inc. Compiled to PTX at build time and
// loaded through the CUDA driver API, so the plugin has no CUDA runtime dependency.
#define DEVICE __device__ __forceinline__
#define PRIV
#define GBUF_CONST const float4*
#define GBUF float4*
#define F4(a, b, c, d) make_float4((a), (b), (c), (d))
#define KERNEL1(name, P) \
    extern "C" __global__ void name(GBUF_CONST src, GBUF dst, P p) { \
        const int x = int(blockIdx.x * blockDim.x + threadIdx.x); \
        const int y = int(blockIdx.y * blockDim.y + threadIdx.y);
#define KERNEL2(name, P) \
    extern "C" __global__ void name(GBUF_CONST src, GBUF_CONST src2, GBUF dst, P p) { \
        const int x = int(blockIdx.x * blockDim.x + threadIdx.x); \
        const int y = int(blockIdx.y * blockDim.y + threadIdx.y);
#define KERNEL_END }

// float4 arithmetic and clamp, built into OpenCL C and Metal.
__device__ __forceinline__ float4 operator+(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
__device__ __forceinline__ float4 operator-(float4 a, float4 b) { return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
__device__ __forceinline__ float4 operator*(float4 a, float k) { return make_float4(a.x * k, a.y * k, a.z * k, a.w * k); }
__device__ __forceinline__ float clamp(float v, float lo, float hi) { return fminf(fmaxf(v, lo), hi); }

#include "tremor_gpu_types.h"
#include "tremor_kernels.inc"
