/* OpenCL C prelude for reception_kernels.inc. */
#define DEVICE
#define GBUF_CONST __global const float4*
#define GBUF __global float4*
#define F4(a, b, c, d) ((float4)((a), (b), (c), (d)))
#define UINT uint
#define RC_PARAMS ReceptionParams
#define FSIN sin
#define FPOW pow
#define FFLOOR floor
#define FMIN fmin
#define FMAX fmax
#define KERNEL(name, P) \
    __kernel void name(GBUF_CONST a, GBUF_CONST b, GBUF_CONST c, GBUF dst, P p) { \
        const int x = (int)get_global_id(0); \
        const int y = (int)get_global_id(1);
#define KERNEL_END }
