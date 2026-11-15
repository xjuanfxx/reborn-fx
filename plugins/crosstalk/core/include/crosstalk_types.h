/* crosstalk_types.h — the parameter block shared by the host, the CPU renderer
 * and the GPU kernels.
 *
 * Compiled as C++ on the host and embedded verbatim in the CUDA, OpenCL and
 * Metal kernel sources, so it must stay plain C: only 4-byte `int`, `unsigned`
 * and `float` fields (no padding), no includes, no host-only types.
 *
 * The "canvas" is the whole layer at the current (possibly downsampled)
 * resolution. Everything that depends on rows, columns or distances is defined
 * on the full-resolution layer, so half and quarter resolution previews look
 * like the final render.
 *
 * Arrays of two hold the settings of the even [0] and odd [1] fields.
 */
#ifndef CROSSTALK_TYPES_H
#define CROSSTALK_TYPES_H

typedef struct CrosstalkParams {
    /* --- geometry -------------------------------------------------------- */
    int canvas_w, canvas_h;      /* canvas size, current resolution */
    int full_w, full_h;          /* layer size, full resolution */
    int in_w, in_h, in_pitch;    /* input buffer; pitch in pixels */
    int in_ox, in_oy;            /* input buffer corner, in canvas coords */
    int out_w, out_h, out_pitch; /* output buffer */
    int out_ox, out_oy;          /* output pixel (0,0), in canvas coords */
    float ds_x, ds_y;            /* current resolution / full resolution */
    float inv_ds_x, inv_ds_y;

    /* --- global ---------------------------------------------------------- */
    unsigned key;                /* seed + frame */
    int edges;                   /* 1 transparent, 2 mirror, 3 tile, 4 black */
    float blend;                 /* Blend With Original, 0–1 */
    float color_max;             /* 1 at 8/16 bpc, very large at 32 bpc */
    float influence;

    /* --- per field: [0] even, [1] odd ------------------------------------- */
    int size[2];                 /* band height, full-resolution rows */
    float process[2];            /* 0–1: original -> tinted */
    float lum[2];                /* brightness offset, 0–1 units */
    int tint[2];
    float tint_r[2], tint_g[2], tint_b[2];
    float noise[2];              /* noise amplitude, 0–1 units */
    int chroma[2];               /* independent noise per channel */
    int decay[2];                /* echo taps */
    int smear[2];
    int method[2];               /* 1 Shift .. 6 Wave Motion */
    float hadj[2];               /* Horiz. Adj. */
    int comb[2];                 /* column mask */
} CrosstalkParams;

#endif
