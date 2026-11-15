/* reception_kernels.inc — the whole effect, written once.
 *
 * The same source runs on the CPU (compiled as C++, see reception_cpu.h) and on
 * the GPU (CUDA, OpenCL and Metal, see plugin/gpu/prelude_*). Each prelude
 * defines:
 *   DEVICE, GBUF, GBUF_CONST, F4, UINT, RC_PARAMS, FSIN, FPOW, FFLOOR, FMIN, FMAX,
 *   KERNEL(name, P) -> body with GBUF_CONST a, b, c, GBUF dst, P p, int x, int y
 *   KERNEL_END.
 *
 * Passes (see docs/behaviour.md):
 *   reception_rows     one thread: the per-row random walks (brightness,
 *                      contrast, line noise, distortion offset) -> row table
 *   reception_distort  per canvas pixel: tint, brightness, contrast and noise
 *                      of the source pixel, displaced by the distortion waves
 *   reception_glow_h / reception_glow_v   thresholded tent blur (Glow)
 *   reception_final    per output pixel: glow (screen), ghosts, bars, vertical
 *                      hold, blend
 *
 * Pixels are float4 in After Effects' GPU layout (x = B, y = G, z = R, w = A),
 * premultiplied. Colour operations work on unpremultiplied 8-bit-style values.
 * The row walks use the Microsoft C runtime rand() sequence, like the original;
 * per-pixel noise comes from a hash of (seed, frame, stream, column, row).
 */

#define RC_S_LINE 1u
#define RC_S_PHASE 2u
#define RC_S_MICRO 3u

/* Microsoft C runtime rand() on state s, and its "(rand() % 10000) / 10000". */
#define RC_RAND(s) ((s) = (s) * 214013u + 2531011u, (int)(((s) >> 16) & 0x7fffu))
#define RC_U(s) ((float)(RC_RAND(s) % 10000) / 10000.0f)

DEVICE UINT rc_mix(UINT v) {
    v ^= v >> 16;
    v *= 0x7feb352du;
    v ^= v >> 15;
    v *= 0x846ca68bu;
    v ^= v >> 16;
    return v;
}

DEVICE UINT rc_hash(UINT key, UINT stream, int a, int b) {
    UINT h = rc_mix(key ^ (stream * 0x9e3779b9u));
    h = rc_mix(h ^ ((UINT)a * 0x85ebca6bu));
    return rc_mix(h ^ ((UINT)b * 0xc2b2ae35u));
}

DEVICE float rc_u01(UINT h) { return (float)(h >> 8) * (1.0f / 16777216.0f); }
DEVICE float rc_s11(UINT h) { return rc_u01(h) * 2.0f - 1.0f; }

DEVICE int rc_full_row(RC_PARAMS p, int row) {
    const int fr = (int)FFLOOR(((float)row + 0.5f) * p.inv_ds_y);
    return fr < p.full_h ? fr : p.full_h - 1;
}
DEVICE int rc_full_col(RC_PARAMS p, int col) { return (int)FFLOOR(((float)col + 0.5f) * p.inv_ds_x); }

DEVICE int rc_clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

DEVICE float4 rc_outside(int edges) {
    return edges == 4 ? F4(0.0f, 0.0f, 0.0f, 1.0f) : F4(0.0f, 0.0f, 0.0f, 0.0f);
}

/* Pixel (x, y) of a canvas-sized buffer, possibly outside the canvas. */
DEVICE float4 rc_fetch(GBUF_CONST buf, int pitch, RC_PARAMS p, int x, int y) {
    const int w = p.canvas_w, h = p.canvas_h;
    if (x < 0 || y < 0 || x >= w || y >= h) {
        if (p.edges == 1 || p.edges == 4) return rc_outside(p.edges);
        if (p.edges == 3) {
            x = ((x % w) + w) % w;
            y = ((y % h) + h) % h;
        } else {
            const int pw = 2 * w, ph = 2 * h;
            x = ((x % pw) + pw) % pw;
            y = ((y % ph) + ph) % ph;
            if (x >= w) x = pw - 1 - x;
            if (y >= h) y = ph - 1 - y;
        }
    }
    return buf[y * pitch + x];
}

/* Input pixel at canvas (x, y), possibly outside the canvas. */
DEVICE float4 rc_input(GBUF_CONST in, RC_PARAMS p, int x, int y) {
    const int w = p.canvas_w, h = p.canvas_h;
    if (x < 0 || y < 0 || x >= w || y >= h) {
        if (p.edges == 1 || p.edges == 4) return rc_outside(p.edges);
        if (p.edges == 3) {
            x = ((x % w) + w) % w;
            y = ((y % h) + h) % h;
        } else {
            const int pw = 2 * w, ph = 2 * h;
            x = ((x % pw) + pw) % pw;
            y = ((y % ph) + ph) % ph;
            if (x >= w) x = pw - 1 - x;
            if (y >= h) y = ph - 1 - y;
        }
    }
    const int ix = x - p.in_ox, iy = y - p.in_oy;
    if (ix < 0 || iy < 0 || ix >= p.in_w || iy >= p.in_h) return F4(0.0f, 0.0f, 0.0f, 0.0f);
    return in[iy * p.in_pitch + ix];
}

/* S-curve contrast on an 8-bit-style value; k in [-127, 127]. Values outside
 * 0–255 (32 bpc) pass through. */
DEVICE float rc_contrast(float v, float k) {
    if (k == 0.0f || v < 0.0f || v > 255.0f) return v;
    float m = v > 127.0f ? 255.0f - v : v;
    float e;
    if (k < 0.0f) {
        if (m < 1.0f) m = 1.0f;
        e = (k + 127.0f) / 127.0f;
    } else {
        e = k >= 127.0f ? 127.0f : 127.0f / (127.0f - k);
    }
    const float y = FMIN(FPOW(m / 127.0f, e) * 127.0f, 255.0f);
    return v < 128.0f ? y : 255.0f - y;
}

/* ------------------------------------------------------------------------- */
/* Row walks: one thread writes (brightness, contrast, line noise, offset)    */
/* for every full-resolution row.                                             */
/* ------------------------------------------------------------------------- */

KERNEL(reception_rows, ReceptionParams)
    if (x != 0 || y != 0) return;
    const int h = p.full_h;
    const float hold = (float)(h / 8);

    /* Brightness and contrast: random walks that hold for a random number of rows. */
    UINT s = p.seed + (UINT)p.frame;
    int cvar = 0, bvar = 0, ccnt = 0, bcnt = 0;
    for (int i = 0; i < 4; ++i) {
        if (ccnt-- < 0) {
            cvar = (int)((float)cvar + RC_U(s) * (2.0f * p.contrast_var) - p.contrast_var);
            if (cvar > 127) cvar = 64;
            if (cvar < -127) cvar = -64;
            ccnt = (int)(RC_U(s) * hold);
        }
        if (bcnt-- < 0) {
            bvar = (int)((float)bvar + RC_U(s) * (2.0f * p.bright_var) - p.bright_var);
            if (bvar > 255) bvar = 128;
            if (bvar < -255) bvar = -128;
            bcnt = (int)(RC_U(s) * hold);
        }
    }
    bvar /= 2;
    for (int r = 0; r < h; ++r) {
        if (ccnt-- < 0) {
            cvar = (int)((float)cvar + RC_U(s) * (2.0f * p.contrast_var) - p.contrast_var);
            ccnt = (int)(RC_U(s) * hold);
        }
        if (bcnt-- < 0) {
            bvar = (int)((float)bvar + RC_U(s) * (2.0f * p.bright_var) - p.bright_var);
            bcnt = (int)(RC_U(s) * hold);
        }
        dst[r].x = (float)rc_clampi((int)((float)bvar + p.lift), -200, 200);
        dst[r].y = (float)rc_clampi((int)((float)cvar + p.contrast), -127, 127);
    }

    /* Line noise amplitude: a walk limited to +-Noise Max. that steps every
     * Noise Tempo/2..Noise Tempo rows (after a warm-up over one frame). */
    const float nmax = p.noise_max, nvar = p.noise_var, tempo = p.noise_tempo;
    s = p.seed + (UINT)p.frame;
    float noff = RC_U(s) * (2.0f * nmax) - nmax;
    int ncnt = 0;
    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1) s = p.seed + (UINT)p.frame;
        for (int r = 0; r < h; ++r) {
            if (ncnt-- < 0) {
                noff = noff + (RC_U(s) * (2.0f * nvar) - nvar);
                ncnt = (int)(RC_U(s) * (tempo - tempo / 2.0f) + tempo / 2.0f);
                noff = FMIN(FMAX(noff, -nmax), nmax);
            }
            if (pass == 1) dst[r].z = noff;
        }
    }

    /* Distortion offset: a per-row walk, new every two frames. */
    s = (UINT)(p.frame / 2);
    int off = 0;
    for (int r = 0; r < h; ++r) {
        off = (int)((float)off + RC_U(s) * (2.0f * p.offset_var) - p.offset_var);
        dst[r].w = (float)off;
    }
KERNEL_END

/* ------------------------------------------------------------------------- */
/* Color, noise and distortion: a = input, b = row table.                     */
/* ------------------------------------------------------------------------- */

KERNEL(reception_distort, ReceptionParams)
    if (x >= p.canvas_w || y >= p.canvas_h) return;
    const int fr = rc_full_row(p, y);
    const float4 row = b[fr];

    int sx = x;
    if (p.dist_on) {
        const float fy = (float)fr;
        const float shift = FSIN(p.ph1 + p.freq1 * fy / p.wave_div) * p.amp1 +
                            FSIN(p.ph2 + p.freq2 * fy / p.wave_div) * p.amp2 + row.w;
        sx = x + (int)(shift * p.ds_x);
    }
    const float4 v = rc_input(a, p, sx, y);
    if (v.w <= 0.0f) {
        dst[y * p.d_pitch + x] = F4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }
    const float k = 255.0f / v.w, hi = p.color_max;
    float cb = v.x * k * p.tint_b, cg = v.y * k * p.tint_g, cr = v.z * k * p.tint_r;

    /* Brightness, then contrast, per row. */
    cb = rc_contrast(FMIN(FMAX(cb + row.x, 0.0f), hi), row.y);
    cg = rc_contrast(FMIN(FMAX(cg + row.x, 0.0f), hi), row.y);
    cr = rc_contrast(FMIN(FMAX(cr + row.x, 0.0f), hi), row.y);

    /* Line noise: segments along the line with one value each, plus micro noise. */
    const int fc = rc_full_col(p, sx);
    const float phase = rc_u01(rc_hash(p.key, RC_S_PHASE, fr, 0)) * p.noise_len;
    const int seg = (int)FFLOOR(((float)fc + phase) / p.noise_len);
    const float n = rc_s11(rc_hash(p.key, RC_S_LINE, seg, fr)) * row.z +
                    rc_s11(rc_hash(p.key, RC_S_MICRO, fc, fr)) * p.micro;
    cb = FMIN(FMAX(cb + n, 0.0f), hi);
    cg = FMIN(FMAX(cg + n, 0.0f), hi);
    cr = FMIN(FMAX(cr + n, 0.0f), hi);

    const float back = v.w / 255.0f;
    dst[y * p.d_pitch + x] = F4(cb * back, cg * back, cr * back, v.w);
KERNEL_END

/* ------------------------------------------------------------------------- */
/* Glow: pixels brighter than the threshold, tent-blurred (edges clamped).    */
/* ------------------------------------------------------------------------- */

DEVICE float4 rc_bright(float4 v, float threshold) {
    if (v.w <= 0.0f) return F4(0.0f, 0.0f, 0.0f, 0.0f);
    const float luma = (0.3f * v.z + 0.59f * v.y + 0.11f * v.x) / v.w;
    return luma > threshold ? v : F4(0.0f, 0.0f, 0.0f, 0.0f);
}

KERNEL(reception_glow_h, ReceptionParams)
    if (x >= p.canvas_w || y >= p.canvas_h) return;
    const int r = p.glow_radius;
    float sb = 0.0f, sg = 0.0f, sr = 0.0f, sa = 0.0f;
    for (int k = -r; k <= r; ++k) {
        const int xx = rc_clampi(x + k, 0, p.canvas_w - 1);
        const float wk = (float)(r + 1 - (k < 0 ? -k : k));
        const float4 v = rc_bright(a[y * p.d_pitch + xx], p.glow_threshold);
        sb += v.x * wk;
        sg += v.y * wk;
        sr += v.z * wk;
        sa += v.w * wk;
    }
    const float norm = 1.0f / (float)((r + 1) * (r + 1));
    dst[y * p.t_pitch + x] = F4(sb * norm, sg * norm, sr * norm, sa * norm);
KERNEL_END

KERNEL(reception_glow_v, ReceptionParams)
    if (x >= p.canvas_w || y >= p.canvas_h) return;
    const int r = p.glow_radius;
    float sb = 0.0f, sg = 0.0f, sr = 0.0f, sa = 0.0f;
    for (int k = -r; k <= r; ++k) {
        const int yy = rc_clampi(y + k, 0, p.canvas_h - 1);
        const float wk = (float)(r + 1 - (k < 0 ? -k : k));
        const float4 v = a[yy * p.t_pitch + x];
        sb += v.x * wk;
        sg += v.y * wk;
        sr += v.z * wk;
        sa += v.w * wk;
    }
    const float norm = 1.0f / (float)((r + 1) * (r + 1));
    dst[y * p.g_pitch + x] = F4(sb * norm, sg * norm, sr * norm, sa * norm);
KERNEL_END

/* ------------------------------------------------------------------------- */
/* Final: a = distorted, b = glow, c = input.                                 */
/* ------------------------------------------------------------------------- */

/* Distorted pixel with the glow screened over it. */
DEVICE float4 rc_glowing(GBUF_CONST d, GBUF_CONST g, RC_PARAMS p, int x, int y) {
    float4 v = rc_fetch(d, p.d_pitch, p, x, y);
    if (p.glow_on) {
        const float4 l = rc_fetch(g, p.g_pitch, p, x, y);
        const float i = p.glow_intensity;
        const float lb = FMIN(l.x * i, 1.0f), lg = FMIN(l.y * i, 1.0f), lr = FMIN(l.z * i, 1.0f);
        const float la = FMIN(l.w * i, 1.0f);
        v = F4(v.x + lb * FMAX(1.0f - v.x, 0.0f), v.y + lg * FMAX(1.0f - v.y, 0.0f),
               v.z + lr * FMAX(1.0f - v.z, 0.0f), v.w + la * FMAX(1.0f - v.w, 0.0f));
    }
    return v;
}

KERNEL(reception_final, ReceptionParams)
    if (x >= p.out_w || y >= p.out_h) return;
    const int cx = x + p.out_ox, cy = y + p.out_oy;

    /* Vertical hold: the picture rolls up by `roll` rows, wrapping around. */
    int ry = cy;
    if (p.roll != 0) ry = ((cy + p.roll) % p.canvas_h + p.canvas_h) % p.canvas_h;

    const float4 c0 = rc_glowing(a, b, p, cx, ry);
    float ob = c0.x, og = c0.y, orr = c0.z;
    if (p.ghost_on) {
        const float4 c1 = rc_glowing(a, b, p, cx + p.ghost_off1, ry);
        const float4 c2 = rc_glowing(a, b, p, cx + p.ghost_off2, ry);
        const float4 c3 = rc_glowing(a, b, p, cx + p.ghost_off3, ry);
        ob = p.w_primary * c0.x + p.w_ghost1 * c1.x + p.w_ghost2 * c2.x + p.w_ghost3 * c3.x;
        og = p.w_primary * c0.y + p.w_ghost1 * c1.y + p.w_ghost2 * c2.y + p.w_ghost3 * c3.y;
        orr = p.w_primary * c0.z + p.w_ghost1 * c1.z + p.w_ghost2 * c2.z + p.w_ghost3 * c3.z;
        const float top = c0.w * (p.color_max / 255.0f);
        ob = FMIN(FMAX(ob, 0.0f), top);
        og = FMIN(FMAX(og, 0.0f), top);
        orr = FMIN(FMAX(orr, 0.0f), top);
    }

    /* Bars: a dark band with a ripple, scrolling with time. */
    if (p.bars_on && p.bar_len > 0.0f) {
        float v = p.bar_pos + (float)rc_full_row(p, ry) * p.bar_step;
        if (v > p.bar_len) v = p.bar_back;
        if (v >= 0.0f) {
            const float gain = FMIN(FMAX(1.0f - ((v / p.bar_len) * p.bar_dark + FSIN(v / 19.0f + p.bar_phase) * 0.3f),
                                         0.0f), 1.0f);
            ob *= gain;
            og *= gain;
            orr *= gain;
        }
    }

    const float4 s = rc_input(c, p, cx, cy);
    const float bl = p.blend, bk = 1.0f - bl;
    dst[y * p.out_pitch + x] = F4(ob * bk + s.x * bl, og * bk + s.y * bl, orr * bk + s.z * bl, c0.w * bk + s.w * bl);
KERNEL_END
