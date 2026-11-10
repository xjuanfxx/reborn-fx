/* tremor_gpu_types.h — parameter blocks shared by the host and the GPU kernels.
 *
 * Compiled as C++ on the host and embedded verbatim in the CUDA, OpenCL and
 * Metal kernel sources, so it must stay plain C: only `int`/`float` fields
 * (4 bytes each, no padding), no includes, no host-only types.
 *
 * Pixels are float4 in After Effects' GPU layout: x = B, y = G, z = R, w = A,
 * premultiplied. Pitches are in pixels.
 */
#ifndef TREMOR_GPU_TYPES_H
#define TREMOR_GPU_TYPES_H

/* Pass 1, per canvas pixel: the input placed on the whole-layer canvas
 * (transparent outside the input), with the field filter. */
typedef struct TremorSourceParams {
    int w, h, dst_pitch;          /* canvas */
    int in_w, in_h, in_pitch;
    int in_ox, in_oy;             /* input buffer corner, in canvas coords */
    int mode;                     /* tremor::Deinterlace */
} TremorSourceParams;

/* Pass 2, per canvas pixel: R, G and B rotated by their own angle. */
typedef struct TremorRotateParams {
    int w, h, src_pitch, dst_pitch;
    int edges;                     /* tremor::Edges */
    float cx, cy;
    float c0, s0, c1, s1, c2, s2;  /* cos/sin of the R, G, B angles */
} TremorRotateParams;

/* Pass 3, per output pixel: zoom + shift, color offsets, composite and blend. */
typedef struct TremorFinishParams {
    int out_w, out_h, out_pitch;
    int w, h, rot_pitch, src_pitch;  /* canvas (rotated and source buffers) */
    int ox, oy;                      /* output pixel (0,0), in canvas coords */
    int edges;
    int composite;
    int zoom_valid;                  /* 0 when the zoom collapses the image (1 + zoom = 0) */
    float cx, cy, inv_zoom;          /* sample = c + (p - c) * inv_zoom - shift */
    float shift_x, shift_y;
    float adj_r, adj_g, adj_b;
    float blend;
    float color_max;
} TremorFinishParams;

#endif
