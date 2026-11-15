/* analog_kernels.inc — the whole effect, written once.
 *
 * The same source runs on the CPU (compiled as C++, see analog_cpu.h) and on
 * the GPU (CUDA, OpenCL and Metal, see plugin/gpu/prelude_*). Each prelude
 * defines:
 *   DEVICE, GBUF, GBUF_CONST, F4, UINT, AN_PARAMS,
 *   FSIN, FCOS, FFLOOR, FABS,
 *   KERNEL1(name, P) -> body with GBUF_CONST src, GBUF dst, P p, int x, int y
 *   KERNEL2(name, P) -> same plus GBUF_CONST src2 (before dst), KERNEL_END.
 *
 * Passes (see docs/behaviour.md):
 *   analog_prepare    per line sample: roll, line jitter, RGB -> YIQ
 *   analog_line_a     per row: composite encode/decode, noise, FM luma, tape
 *   analog_line_b     per row: vertical chroma blend, sharpen, re-modulation,
 *                     chroma loss, extra generations, output low-pass
 *   analog_output     per output pixel: YIQ -> RGB, alpha
 *   analog_blur_down / analog_blur_up   optional scanline blur
 *
 * Everything random comes from a counter-based hash of (seed, frame, stream,
 * row, sample): no state, identical on every thread, device and resolution.
 *
 * Line buffers hold float4 (Y, I, Q, unused) in signal units (white = 255).
 */

#define AN_PI 3.14159265f
#define AN_TWO_PI 6.28318531f
#define AN_FM_LO (-195.3125f) /* FM input/output range, signal units */
#define AN_FM_SPAN 585.9375f

/* Random streams. */
#define AN_S_DIST1 1u
#define AN_S_DIST2 2u
#define AN_S_HUE 3u
#define AN_S_LOSS 4u
#define AN_S_NOISE_Y 5u
#define AN_S_NOISE_I 6u
#define AN_S_NOISE_Q 7u
#define AN_S_TE_GATE 8u
#define AN_S_TE_A 9u
#define AN_S_TE_B 10u
#define AN_S_TE_C 11u

#define AN_AT(p, xs, row) ((xs) * (p).plane_sx + (row) * (p).plane_sy)

/* ------------------------------------------------------------------------- */
/* Random numbers                                                             */
/* ------------------------------------------------------------------------- */

DEVICE UINT an_mix(UINT v) {
    v ^= v >> 16;
    v *= 0x7feb352du;
    v ^= v >> 15;
    v *= 0x846ca68bu;
    v ^= v >> 16;
    return v;
}

DEVICE UINT an_hash(UINT key, UINT stream, int a, int b) {
    UINT h = an_mix(key ^ (stream * 0x9e3779b9u));
    h = an_mix(h ^ ((UINT)a * 0x85ebca6bu));
    return an_mix(h ^ ((UINT)b * 0xc2b2ae35u));
}

/* [0, 1) and [-1, 1), exact in float. */
DEVICE float an_u01(UINT h) { return (float)(h >> 8) * (1.0f / 16777216.0f); }
DEVICE float an_s11(UINT h) { return an_u01(h) * 2.0f - 1.0f; }

/* ------------------------------------------------------------------------- */
/* Per-row values                                                             */
/* ------------------------------------------------------------------------- */

/* Full-resolution row shown by canvas row `row` (keys every per-row random). */
DEVICE int an_full_row(AN_PARAMS p, int row) {
    return (int)FFLOOR(((float)row + 0.5f) * p.inv_ds_y);
}

/* Subcarrier phase (quarter cycles) at the start of a line. */
DEVICE int an_row_phase(AN_PARAMS p, int fr) {
    const UINT r = (UINT)fr;
    int ph = p.phase_offset;
    if (p.phase_mode == 1) ph += (int)(r >> 1);
    else if (p.phase_mode == 2) ph += (int)(r & 2u);
    else if (p.phase_mode == 3) ph -= (int)(r >> 1);
    return ph & 3;
}

/* Subcarrier at four samples per cycle. */
DEVICE float an_cos4(int k) {
    k &= 3;
    return k == 0 ? 1.0f : (k == 2 ? -1.0f : 0.0f);
}
DEVICE float an_sin4(int k) {
    k &= 3;
    return k == 1 ? 1.0f : (k == 3 ? -1.0f : 0.0f);
}

/* A per-line random value in [-1, 1): new every frame, or drifting smoothly
 * through time at dist_rate values per second. */
DEVICE float an_dist_value(AN_PARAMS p, UINT stream, int fr) {
    if (p.dist_rate <= 0.0f) return an_s11(an_hash(p.key, stream, fr, 0));
    const float tk = p.time_s * p.dist_rate;
    const float k = FFLOOR(tk);
    float f = tk - k;
    const int ki = (int)k;
    const float a = an_s11(an_hash(p.seed_key, stream, fr, ki));
    const float b = an_s11(an_hash(p.seed_key, stream, fr, ki + 1));
    f = f * f * (3.0f - 2.0f * f);
    return a + (b - a) * f;
}

/* Horizontal line jitter (TV distortion), whole full-resolution samples. */
DEVICE float an_shift(AN_PARAMS p, int fr) {
    if (p.dist_thick <= 0.0f && p.dist_fine <= 0.0f) return 0.0f;
    float s1 = an_dist_value(p, AN_S_DIST1, fr) * p.dist_thick;
    s1 += (s1 <= 0.0f ? -0.1f : 0.1f) * p.dist_thick;
    float s2 = an_dist_value(p, AN_S_DIST2, fr) * p.dist_fine;
    s2 += (s2 <= 0.0f ? -0.1f : 0.1f) * p.dist_fine;
    return (float)(int)(s1 + s2);
}

/* Hue error of a line: a random walk down the picture (each step halves the
 * previous error), truncated after 24 lines so every row stands alone. */
DEVICE float an_hue_angle(AN_PARAMS p, int fr) {
    float w = 0.0f, wt = 0.5f;
    for (int k = 0; k < 24; ++k) {
        w += wt * an_s11(an_hash(p.key, AN_S_HUE, fr - k, 0));
        wt *= 0.5f;
    }
    return w * p.hue_noise * (3.14f / 100.0f);
}

/* ------------------------------------------------------------------------- */
/* Line operations (one row of the line buffer)                               */
/* ------------------------------------------------------------------------- */

/* Puts I and Q on the subcarrier and adds it to Y: Y becomes the composite. */
DEVICE void an_modulate(GBUF L, AN_PARAMS p, int row, int ph) {
    for (int xs = 0; xs < p.line_w; ++xs) {
        const int i = AN_AT(p, xs, row);
        float4 v = L[i];
        const int k = ph + xs;
        v.x += (v.y * an_cos4(k) + v.z * an_sin4(k)) * p.amp;
        v.y = 0.0f;
        v.z = 0.0f;
        L[i] = v;
    }
}

/* Splits the composite back into Y, I and Q. Luma is the composite through a
 * [1 2 2 2 1]/8 notch (zero at the subcarrier); the rest is chroma, which
 * holds I on even phases and Q on odd ones; the missing samples are the mean
 * of their neighbours. */
DEVICE void an_demodulate(GBUF L, AN_PARAMS p, int row, int ph) {
    const int n = p.line_w;
    const int last = n - 1;
    const float c0 = L[AN_AT(p, 0, row)].x;
    float wm2 = c0, wm1 = c0, w0 = c0;
    float w1 = L[AN_AT(p, last < 1 ? last : 1, row)].x;
    float w2 = L[AN_AT(p, last < 2 ? last : 2, row)].x;
    for (int xs = 0; xs < n; ++xs) {
        const float lum = (wm2 + 2.0f * (wm1 + w0 + w1) + w2) * 0.125f;
        const float c = (w0 - lum) * p.inv_amp;
        const int k = ph + xs;
        L[AN_AT(p, xs, row)] = F4(lum, c * an_cos4(k), c * an_sin4(k), 0.0f);
        wm2 = wm1;
        wm1 = w0;
        w0 = w1;
        w1 = w2;
        const int nx = xs + 3;
        w2 = L[AN_AT(p, nx < last ? nx : last, row)].x;
    }
    for (int xs = 0; xs < n; ++xs) {
        const int i = AN_AT(p, xs, row);
        const int l = xs > 0 ? xs - 1 : (xs < last ? xs + 1 : xs);
        const int r = xs < last ? xs + 1 : (xs > 0 ? xs - 1 : xs);
        const float4 a = L[AN_AT(p, l, row)];
        const float4 b = L[AN_AT(p, r, row)];
        float4 v = L[i];
        if (((ph + xs) & 1) == 0) v.z = 0.5f * (a.z + b.z);
        else v.y = 0.5f * (a.y + b.y);
        L[i] = v;
    }
}

/* Three one-pole low-passes on I and Q, moved back by si / sq samples to
 * compensate their delay (or forward, for the tape's chroma delay). */
DEVICE void an_lp3_iq(GBUF L, AN_PARAMS p, int row, float ai, float aq, int si, int sq) {
    const int n = p.line_w;
    const float4 v0 = L[AN_AT(p, 0, row)];
    float i1 = v0.y, i2 = v0.y, i3 = v0.y;
    float q1 = v0.z, q2 = v0.z, q3 = v0.z;
    for (int xs = 0; xs < n; ++xs) {
        const float4 v = L[AN_AT(p, xs, row)];
        i1 += ai * (v.y - i1);
        i2 += ai * (i1 - i2);
        i3 += ai * (i2 - i3);
        q1 += aq * (v.z - q1);
        q2 += aq * (q1 - q2);
        q3 += aq * (q2 - q3);
        if (xs >= si) L[AN_AT(p, xs - si, row)].y = i3;
        if (xs >= sq) L[AN_AT(p, xs - sq, row)].z = q3;
    }
    for (int xs = (n - si > 0 ? n - si : 0); xs < n; ++xs) L[AN_AT(p, xs, row)].y = i3;
    for (int xs = (n - sq > 0 ? n - sq : 0); xs < n; ++xs) L[AN_AT(p, xs, row)].z = q3;
}

/* Boosts the high frequencies of the composite (Signal Amplification). */
DEVICE void an_preemphasis(GBUF L, AN_PARAMS p, int row) {
    float s = L[AN_AT(p, 0, row)].x;
    for (int xs = 0; xs < p.line_w; ++xs) {
        const int i = AN_AT(p, xs, row);
        float4 v = L[i];
        s += p.a_pre * (v.x - s);
        v.x += (v.x - s) * p.pre_amount;
        L[i] = v;
    }
}

/* Low-frequency noise on the composite: each step halves the previous error
 * and adds a new random one. */
DEVICE void an_luma_noise(GBUF L, AN_PARAMS p, int row, int fr) {
    float walk = 0.0f;
    for (int xs = 0; xs < p.line_w; ++xs) {
        const int i = AN_AT(p, xs, row);
        float4 v = L[i];
        v.x += walk;
        walk = 0.5f * (walk + p.noise_luma * an_s11(an_hash(p.key, AN_S_NOISE_Y, fr, xs)));
        L[i] = v;
    }
}

DEVICE void an_chroma_noise(GBUF L, AN_PARAMS p, int row, int fr) {
    float wi = 0.0f, wq = 0.0f;
    for (int xs = 0; xs < p.line_w; ++xs) {
        const int i = AN_AT(p, xs, row);
        float4 v = L[i];
        v.y += wi;
        v.z += wq;
        wi = 0.5f * (wi + p.noise_chroma * an_s11(an_hash(p.key, AN_S_NOISE_I, fr, xs)));
        wq = 0.5f * (wq + p.noise_chroma * an_s11(an_hash(p.key, AN_S_NOISE_Q, fr, xs)));
        L[i] = v;
    }
}

DEVICE void an_hue_rotate(GBUF L, AN_PARAMS p, int row, float angle) {
    const float c = FCOS(angle), s = FSIN(angle);
    for (int xs = 0; xs < p.line_w; ++xs) {
        const int i = AN_AT(p, xs, row);
        float4 v = L[i];
        const float ci = v.y, cq = v.z;
        v.y = ci * c - cq * s;
        v.z = cq * c + ci * s;
        L[i] = v;
    }
}

/* VHS luma: frequency-modulated onto a carrier, read back with a slope
 * detector (|difference| of the carrier, low-passed). Tape errors replace the
 * detector output with bright spikes in a band near the bottom. */
DEVICE void an_fm(GBUF L, AN_PARAMS p, int row, int fr) {
    const float dev = p.fm_dev, freq = p.fm_freq;
    const float level = freq * p.fm_light;
    const float q = (float)p.fm_quantize;
    const float a = p.a_fm;
    const float frf = (float)fr;
    const int band = p.tape_errors && p.te_y1 > p.te_y0 && frf > p.te_y0 && frf < p.te_y1;
    float te_w = 0.0f;
    if (band) {
        const float t = (frf - p.te_y0) / (p.te_y1 - p.te_y0);
        te_w = (FSIN(t * 4.71f - 1.57f) + 1.0f) * 0.5f * p.te_fade;
    }
    float ph = -freq, prev = 0.0f, f1 = 0.0f, f2 = 0.0f, f3 = 0.0f;
    for (int xs = 0; xs < p.line_w; ++xs) {
        const int i = AN_AT(p, xs, row);
        float4 v = L[i];
        ph += freq + ((v.x - AN_FM_LO) / AN_FM_SPAN) * 2.0f * dev - dev;
        ph -= AN_TWO_PI * FFLOOR((ph + AN_PI) / AN_TWO_PI);
        float s = FSIN(ph);
        if (q > 0.0f) s = FFLOOR((s + 1.0f) * 0.5f * q) / q * 2.0f - 1.0f;
        float d = FABS(s - prev);
        prev = s;
        if (band && an_u01(an_hash(p.key, AN_S_TE_GATE, fr, xs)) < p.te_prob &&
            an_u01(an_hash(p.key, AN_S_TE_A, fr, xs)) < te_w * an_u01(an_hash(p.key, AN_S_TE_B, fr, xs)))
            d = p.te_gamma * an_u01(an_hash(p.key, AN_S_TE_C, fr, xs));
        f1 += a * (d - f1);
        f2 += a * (f1 - f2);
        f3 += a * (f2 - f3);
        v.x = ((2.0f * (f3 - level) + dev) / (2.0f * dev)) * AN_FM_SPAN + AN_FM_LO;
        L[i] = v;
    }
}

/* VHS luma bandwidth: three low-passes, then the tape's high-frequency boost. */
DEVICE void an_vhs_luma(GBUF L, AN_PARAMS p, int row) {
    const float y0 = L[AN_AT(p, 0, row)].x;
    float s1 = y0, s2 = y0, s3 = y0, sl = y0;
    for (int xs = 0; xs < p.line_w; ++xs) {
        const int i = AN_AT(p, xs, row);
        float4 v = L[i];
        s1 += p.a_luma * (v.x - s1);
        s2 += p.a_luma * (s1 - s2);
        s3 += p.a_luma * (s2 - s3);
        sl += p.a_luma_slow * (s3 - sl);
        v.x = s3 + 1.6f * (s3 - sl);
        L[i] = v;
    }
}

DEVICE void an_sharpen(GBUF L, AN_PARAMS p, int row) {
    const float y0 = L[AN_AT(p, 0, row)].x;
    float s1 = y0, s2 = y0, s3 = y0;
    for (int xs = 0; xs < p.line_w; ++xs) {
        const int i = AN_AT(p, xs, row);
        float4 v = L[i];
        s1 += p.a_sharpen * (v.x - s1);
        s2 += p.a_sharpen * (s1 - s2);
        s3 += p.a_sharpen * (s2 - s3);
        v.x += 2.0f * (v.x - s3) * p.sharpen_amount;
        L[i] = v;
    }
}

DEVICE void an_kill_chroma(GBUF L, AN_PARAMS p, int row) {
    for (int xs = 0; xs < p.line_w; ++xs) {
        const int i = AN_AT(p, xs, row);
        float4 v = L[i];
        v.y = 0.0f;
        v.z = 0.0f;
        L[i] = v;
    }
}

/* ------------------------------------------------------------------------- */
/* Output helpers                                                             */
/* ------------------------------------------------------------------------- */

DEVICE float4 an_fetch(GBUF_CONST img, AN_PARAMS p, int cx, int cy) {
    const int bx = cx - p.in_ox, by = cy - p.in_oy;
    if (bx < 0 || by < 0 || bx >= p.in_w || by >= p.in_h) return F4(0.0f, 0.0f, 0.0f, 0.0f);
    return img[by * p.in_pitch + bx];
}

/* Signal units -> output channel: clamp, or wrap around like an 8-bit
 * overflow (Destruction). Premultiplied: never above alpha. */
DEVICE float an_channel(float v, float alpha, AN_PARAMS p) {
    float c;
    if (p.destruction) {
        c = (float)(((int)v) & 255) * (1.0f / 255.0f);
    } else {
        c = v * (1.0f / 255.0f);
        c = c < 0.0f ? 0.0f : (c > p.color_max ? p.color_max : c);
    }
    if (alpha < 1.0f && c > alpha) c = alpha;
    return c;
}

/* ------------------------------------------------------------------------- */
/* Kernels                                                                    */
/* ------------------------------------------------------------------------- */

/* Per line sample (x = sample, y = canvas row): the source picture after roll
 * and line jitter, converted to YIQ. */
KERNEL1(analog_prepare, AnalogParams)
    if (x >= p.line_w || y >= p.rows) return;
    const int fr = an_full_row(p, y);
    int sy = (y - p.roll) % p.rows;
    if (sy < 0) sy += p.rows;
    float xs = (float)x - an_shift(p, fr);
    xs -= (float)p.line_w * FFLOOR(xs / (float)p.line_w);
    const float cxf = (xs + 0.5f) * p.ds_x - 0.5f;
    const float fx = FFLOOR(cxf);
    const float t = cxf - fx;
    int x0 = (int)fx, x1 = x0 + 1;
    x0 = x0 < 0 ? 0 : (x0 >= p.canvas_w ? p.canvas_w - 1 : x0);
    x1 = x1 < 0 ? 0 : (x1 >= p.canvas_w ? p.canvas_w - 1 : x1);
    const float4 c0 = an_fetch(src, p, x0, sy);
    const float4 c1 = an_fetch(src, p, x1, sy);
    const float b = (c0.x + (c1.x - c0.x) * t) * 255.0f;
    const float g = (c0.y + (c1.y - c0.y) * t) * 255.0f;
    const float r = (c0.z + (c1.z - c0.z) * t) * 255.0f;
    const float lum = 0.30f * r + 0.59f * g + 0.11f * b;
    const float ry = r - lum, by = b - lum;
    dst[AN_AT(p, x, y)] = F4(lum, 0.74f * ry - 0.27f * by, 0.48f * ry + 0.41f * by, 0.0f);
KERNEL_END

/* Per row (x = row): composite encode/decode, noise, FM luma and the tape. */
KERNEL1(analog_line_a, AnalogParams)
    (void)src;
    if (x >= p.rows || y > 0) return;
    const int row = x;
    const int fr = an_full_row(p, row);
    const int ph = an_row_phase(p, fr);
    if (p.lp_in) an_lp3_iq(dst, p, row, p.a_lpin_i, p.a_lpin_q, 2, 4);
    an_modulate(dst, p, row, ph);
    if (p.pre_amount != 0.0f && p.a_pre > 0.0f) an_preemphasis(dst, p, row);
    if (p.noise_luma > 0.0f) an_luma_noise(dst, p, row, fr);
    an_demodulate(dst, p, row, ph);
    if (p.fm) an_fm(dst, p, row, fr);
    if (p.noise_chroma > 0.0f) an_chroma_noise(dst, p, row, fr);
    if (p.hue_noise > 0.0f) an_hue_rotate(dst, p, row, an_hue_angle(p, fr));
    if (p.vhs) {
        an_vhs_luma(dst, p, row);
        an_lp3_iq(dst, p, row, p.a_chroma, p.a_chroma, p.chroma_delay, p.chroma_delay);
    }
KERNEL_END

/* Per row (x = row): reads the line_a result of this row and the one above,
 * writes the final line. */
KERNEL1(analog_line_b, AnalogParams)
    if (x >= p.rows || y > 0) return;
    const int row = x;
    const int fr = an_full_row(p, row);
    const int ph = an_row_phase(p, fr);
    const int above = row > 0 ? row - 1 : row;
    for (int xs = 0; xs < p.line_w; ++xs) {
        float4 v = src[AN_AT(p, xs, row)];
        if (p.vblend) {
            const float4 u = src[AN_AT(p, xs, above)];
            v.y = 0.5f * (v.y + u.y);
            v.z = 0.5f * (v.z + u.z);
        }
        dst[AN_AT(p, xs, row)] = v;
    }
    if (p.sharpen && p.sharpen_amount != 0.0f) an_sharpen(dst, p, row);
    if (!p.svideo) {
        an_modulate(dst, p, row, ph);
        an_demodulate(dst, p, row, ph);
    }
    if (p.chroma_loss > 0.0f && an_u01(an_hash(p.key, AN_S_LOSS, fr, 0)) < p.chroma_loss)
        an_kill_chroma(dst, p, row);
    for (int g = 0; g < p.generations; ++g) {
        an_modulate(dst, p, row, ph);
        an_demodulate(dst, p, row, ph);
    }
    if (p.lp_out == 1) an_lp3_iq(dst, p, row, p.a_lpout_lite, p.a_lpout_lite, 1, 1);
    else if (p.lp_out == 2) an_lp3_iq(dst, p, row, p.a_lpin_i, p.a_lpin_q, 2, 4);
KERNEL_END

/* Per output pixel: mean of the line samples it covers, YIQ -> RGB, and the
 * source alpha at the same place. src = line buffer, src2 = input. */
KERNEL2(analog_output, AnalogParams)
    if (x >= p.out_w || y >= p.out_h) return;
    const int cx = x + p.out_ox, cy = y + p.out_oy;
    float4 o = F4(0.0f, 0.0f, 0.0f, 0.0f);
    if (cx >= 0 && cy >= 0 && cx < p.canvas_w && cy < p.rows) {
        int x0 = (int)FFLOOR((float)cx * p.inv_ds_x);
        int x1 = (int)FFLOOR((float)(cx + 1) * p.inv_ds_x);
        if (x0 > p.line_w - 1) x0 = p.line_w - 1;
        if (x1 > p.line_w) x1 = p.line_w;
        if (x1 <= x0) x1 = x0 + 1;
        float ly = 0.0f, li = 0.0f, lq = 0.0f;
        for (int xs = x0; xs < x1; ++xs) {
            const float4 v = src[AN_AT(p, xs, cy)];
            ly += v.x;
            li += v.y;
            lq += v.z;
        }
        const float inv = 1.0f / (float)(x1 - x0);
        ly *= inv;
        li *= inv;
        lq *= inv;
        const float alpha = an_fetch(src2, p, cx, cy).w;
        const float r = an_channel(ly + 0.956f * li + 0.621f * lq, alpha, p);
        const float g = an_channel(ly - 0.272f * li - 0.647f * lq, alpha, p);
        const float b = an_channel(ly - 1.106f * li + 1.703f * lq, alpha, p);
        o = F4(b, g, r, alpha);
    }
    dst[y * p.out_pitch + x] = o;
KERNEL_END

/* Scanline blur, step 1: area-average the full canvas (src, tmp_pitch) down to
 * blur_w x blur_h (dst, small_pitch). */
KERNEL1(analog_blur_down, AnalogParams)
    if (x >= p.blur_w || y >= p.blur_h) return;
    const int x0 = x * p.canvas_w / p.blur_w;
    int x1 = (x + 1) * p.canvas_w / p.blur_w;
    const int y0 = y * p.rows / p.blur_h;
    int y1 = (y + 1) * p.rows / p.blur_h;
    if (x1 <= x0) x1 = x0 + 1;
    if (y1 <= y0) y1 = y0 + 1;
    float4 s = F4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int yy = y0; yy < y1; ++yy)
        for (int xx = x0; xx < x1; ++xx) {
            const float4 v = src[yy * p.tmp_pitch + xx];
            s.x += v.x;
            s.y += v.y;
            s.z += v.z;
            s.w += v.w;
        }
    const float inv = 1.0f / (float)((x1 - x0) * (y1 - y0));
    const float aw = s.w * inv;
    dst[y * p.small_pitch + x] = F4(s.x * inv, s.y * inv, s.z * inv, aw > 1.0f ? 1.0f : aw);
KERNEL_END

/* Scanline blur, step 2: bilinear back up into the output. */
KERNEL1(analog_blur_up, AnalogParams)
    if (x >= p.out_w || y >= p.out_h) return;
    const int cx = x + p.out_ox, cy = y + p.out_oy;
    float4 o = F4(0.0f, 0.0f, 0.0f, 0.0f);
    if (cx >= 0 && cy >= 0 && cx < p.canvas_w && cy < p.rows) {
        float u = ((float)cx + 0.5f) * (float)p.blur_w / (float)p.canvas_w - 0.5f;
        float v = ((float)cy + 0.5f) * (float)p.blur_h / (float)p.rows - 0.5f;
        u = u < 0.0f ? 0.0f : (u > (float)(p.blur_w - 1) ? (float)(p.blur_w - 1) : u);
        v = v < 0.0f ? 0.0f : (v > (float)(p.blur_h - 1) ? (float)(p.blur_h - 1) : v);
        const int u0 = (int)u, v0 = (int)v;
        const int u1 = u0 + 1 < p.blur_w ? u0 + 1 : u0;
        const int v1 = v0 + 1 < p.blur_h ? v0 + 1 : v0;
        const float tu = u - (float)u0, tv = v - (float)v0;
        const float4 a = src[v0 * p.small_pitch + u0], b = src[v0 * p.small_pitch + u1];
        const float4 c = src[v1 * p.small_pitch + u0], d = src[v1 * p.small_pitch + u1];
        const float w00 = (1.0f - tu) * (1.0f - tv), w10 = tu * (1.0f - tv);
        const float w01 = (1.0f - tu) * tv, w11 = tu * tv;
        o = F4(a.x * w00 + b.x * w10 + c.x * w01 + d.x * w11, a.y * w00 + b.y * w10 + c.y * w01 + d.y * w11,
               a.z * w00 + b.z * w10 + c.z * w01 + d.z * w11, a.w * w00 + b.w * w10 + c.w * w01 + d.w * w11);
        /* Averaging can leave alpha a rounding error above 1. */
        o.w = o.w > 1.0f ? 1.0f : o.w;
        if (o.w < 1.0f) {
            o.x = o.x > o.w ? o.w : o.x;
            o.y = o.y > o.w ? o.w : o.y;
            o.z = o.z > o.w ? o.w : o.z;
        }
    }
    dst[y * p.out_pitch + x] = o;
KERNEL_END
