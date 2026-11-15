/* reception_types.h — the parameter block shared by the host, the CPU renderer
 * and the GPU kernels.
 *
 * Compiled as C++ on the host and embedded verbatim in the CUDA, OpenCL and
 * Metal kernel sources, so it must stay plain C: only 4-byte `int`, `unsigned`
 * and `float` fields (no padding), no includes, no host-only types.
 *
 * The "canvas" is the whole layer at the current (possibly downsampled)
 * resolution. Rows, widths and distances are defined on the full-resolution
 * layer, so half and quarter resolution previews look like the final render.
 * Colour values are handled in 8-bit units (white = 255) like the original.
 */
#ifndef RECEPTION_TYPES_H
#define RECEPTION_TYPES_H

typedef struct ReceptionParams {
    /* --- geometry -------------------------------------------------------- */
    int canvas_w, canvas_h;      /* canvas size, current resolution */
    int full_w, full_h;          /* layer size, full resolution */
    int in_w, in_h, in_pitch;    /* input buffer; pitch in pixels */
    int in_ox, in_oy;            /* input buffer corner, in canvas coords */
    int out_w, out_h, out_pitch; /* output buffer */
    int out_ox, out_oy;          /* output pixel (0,0), in canvas coords */
    int d_pitch, t_pitch, g_pitch; /* scratch buffers (canvas size): distorted, glow pass 1, glow */
    float ds_x, ds_y;            /* current resolution / full resolution */
    float inv_ds_x, inv_ds_y;

    /* --- global ---------------------------------------------------------- */
    int frame;                   /* layer frame number */
    unsigned seed;               /* Random Seed */
    unsigned key;                /* hash of seed and frame (per-pixel noise) */
    int edges;                   /* 1 transparent, 2 mirror, 3 tile, 4 black */
    float blend;                 /* Blend With Original, 0–1 */
    float color_max;             /* 255 at 8/16 bpc, very large at 32 bpc */

    /* --- color (row walks in reception_rows) ----------------------------- */
    float tint_r, tint_g, tint_b; /* 0–1 multipliers */
    float lift;                  /* brightness offset, 8-bit levels */
    float bright_var;            /* brightness walk step */
    float contrast;              /* Contrast x 2.55 */
    float contrast_var;          /* contrast walk step */

    /* --- noise ------------------------------------------------------------- */
    float noise_max;             /* line noise limit, 8-bit levels */
    float noise_var;             /* line noise walk step */
    float noise_tempo;           /* rows between line noise steps */
    float micro;                 /* per-pixel noise, 8-bit levels */
    float noise_len;             /* noise segment length, full-resolution pixels */

    /* --- distortion -------------------------------------------------------- */
    int dist_on;
    float amp1, amp2;            /* wave amplitudes this frame, full-resolution pixels */
    float ph1, ph2;              /* wave phases this frame */
    float freq1, freq2;
    float wave_div;              /* full height x 0.1 + 1 */
    float offset_var;            /* per-row offset walk step, full-resolution pixels */

    /* --- glow -------------------------------------------------------------- */
    int glow_on;
    int glow_radius;             /* tent radius, canvas pixels */
    float glow_threshold;        /* luma threshold, 0–1 */
    float glow_intensity;

    /* --- ghosting ---------------------------------------------------------- */
    int ghost_on;
    int ghost_off1, ghost_off2, ghost_off3; /* canvas pixels */
    float w_primary, w_ghost1, w_ghost2, w_ghost3;

    /* --- bars -------------------------------------------------------------- */
    int bars_on;
    float bar_pos;               /* band position at row 0 */
    float bar_step;              /* per full-resolution row */
    float bar_back;              /* restart position (-L) */
    float bar_len;
    float bar_dark;              /* 0–1 */
    float bar_phase;             /* frame / 40 */

    /* --- vertical hold ----------------------------------------------------- */
    int roll;                    /* canvas rows */
} ReceptionParams;

#endif
