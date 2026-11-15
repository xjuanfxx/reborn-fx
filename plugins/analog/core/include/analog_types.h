/* analog_types.h — the parameter block shared by the host, the CPU renderer and
 * the GPU kernels.
 *
 * Compiled as C++ on the host and embedded verbatim in the CUDA, OpenCL and
 * Metal kernel sources, so it must stay plain C: only 4-byte `int`, `unsigned`
 * and `float` fields (no padding), no includes, no host-only types.
 *
 * Coordinates: the "canvas" is the whole layer at the current (possibly
 * downsampled) resolution. Every line is processed at full resolution — the
 * "line buffer" has `line_w` samples per canvas row — so previews at half or
 * quarter resolution look like the final render.
 *
 * Signal units: luma and chroma are carried in 0–255 "IRE-like" units
 * (white = 255) while the line is processed.
 */
#ifndef ANALOG_TYPES_H
#define ANALOG_TYPES_H

typedef struct AnalogParams {
    /* --- geometry ------------------------------------------------------ */
    int canvas_w, rows;          /* canvas size, current resolution */
    int line_w;                  /* samples per line (full resolution) */
    int full_h;                  /* canvas rows at full resolution */
    int plane_sx, plane_sy;      /* line-buffer index = xs * plane_sx + row * plane_sy */
    int in_w, in_h, in_pitch;    /* input buffer; pitch in pixels */
    int in_ox, in_oy;            /* input buffer corner, in canvas coords */
    int out_w, out_h, out_pitch; /* output buffer */
    int out_ox, out_oy;          /* output pixel (0,0), in canvas coords */
    float ds_x, ds_y;            /* current resolution / full resolution */
    float inv_ds_x, inv_ds_y;
    int roll;                    /* vertical roll, canvas rows */

    /* --- randomness ----------------------------------------------------- */
    unsigned key;                /* seed + frame: changes every frame */
    unsigned seed_key;           /* seed only: time-smooth distortion */
    float time_s;                /* layer time, seconds */

    /* --- composite signal ------------------------------------------------- */
    int lp_in;                   /* chroma low-pass before modulation */
    int lp_out;                  /* 0 off, 1 lite, 2 strong */
    int phase_mode;              /* subcarrier phase per line: 0, 90, 180, 270 degrees */
    int phase_offset;            /* 0..3 quarter cycles */
    float amp;                   /* chroma subcarrier amplitude (Signal Strength / 50) */
    float inv_amp;               /* 1 / amp, 0 when amp is 0 */
    float pre_amount;            /* pre-emphasis (Signal Amplification) */
    float a_pre;                 /* pre-emphasis one-pole coefficient */
    float a_lpin_i, a_lpin_q;    /* chroma low-pass (I 1.3 MHz, Q 600 kHz) */
    float a_lpout_lite;          /* chroma low-pass 2.6 MHz */
    float noise_luma;            /* composite noise amplitude (signal units) */
    float noise_chroma;          /* chroma noise amplitude (signal units) */
    float hue_noise;             /* hue walk amplitude */
    float chroma_loss;           /* probability that a line loses its color */
    int generations;             /* extra modulate/demodulate passes */
    int svideo;                  /* 1: no re-modulation after the tape */

    /* --- tape ------------------------------------------------------------- */
    int vhs;                     /* 0 off, 1 SP, 2 LP, 3 EP */
    int chroma_delay;            /* samples */
    float a_luma, a_luma_slow;   /* VHS luma low-pass and its slow high-pass */
    float a_chroma;              /* VHS chroma low-pass */
    int vblend;                  /* chroma vertical blend */
    int sharpen;                 /* sharpen on/off */
    float sharpen_amount;
    float a_sharpen;

    /* --- FM luma ("Luma modulation") ---------------------------------- */
    int fm;                      /* on/off */
    int fm_quantize;             /* 0 = off, else levels */
    float fm_freq;               /* carrier, radians per sample */
    float fm_dev;                /* deviation (radians per sample) */
    float fm_light;              /* demodulator level */
    float a_fm;                  /* demodulator low-pass */
    int tape_errors;
    float te_y0, te_y1;          /* band, full-resolution rows */
    float te_prob, te_fade, te_gamma;

    /* --- TV distortion ------------------------------------------------------- */
    float dist_thick, dist_fine; /* per-line horizontal jitter, full-resolution pixels */
    float dist_rate;             /* new random values per second; 0 = every frame */

    /* --- output -------------------------------------------------------------- */
    int destruction;             /* wrap instead of clamp */
    float color_max;             /* 1 at 8/16 bpc, very large at 32 bpc */
    float blur;                  /* scanline blur factor (full-resolution pixels), <= 1 off */
    int blur_w, blur_h;          /* low-resolution size for the blur */
    int tmp_pitch, small_pitch;  /* blur buffers (full canvas, low resolution) */
} AnalogParams;

#endif
