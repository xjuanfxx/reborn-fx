/* crosstalk_kernels.inc — the whole effect, written once.
 *
 * The same source runs on the CPU (compiled as C++, see crosstalk_cpu.h) and on
 * the GPU (CUDA, OpenCL and Metal, see plugin/gpu/prelude_*). Each prelude
 * defines:
 *   DEVICE, GBUF, GBUF_CONST, F4, UINT, CT_PARAMS, FSIN, FFLOOR, FMIN, FMAX,
 *   KERNEL(name, P) -> body with GBUF_CONST a, b, c, GBUF dst, P p, int x, int y
 *   KERNEL_END.
 *
 * One pass, per output pixel (see docs/behaviour.md):
 *   field of the row -> horizontal displacement -> tint and Process Input ->
 *   luminance, noise and echo (Decay / Smear) -> comb filter -> blend.
 *
 * Pixels are float4 in After Effects' GPU layout (x = B, y = G, z = R, w = A),
 * premultiplied. Colour operations work on unpremultiplied values.
 * Everything random comes from a hash of (seed, frame, stream, column, row):
 * no state, identical on every thread, device and resolution.
 */

#define CT_S_NOISE 1u
#define CT_S_NOISE_G 2u
#define CT_S_NOISE_B 3u
#define CT_S_TAP 4u
#define CT_S_ROW 5u

DEVICE UINT ct_mix(UINT v) {
    v ^= v >> 16;
    v *= 0x7feb352du;
    v ^= v >> 15;
    v *= 0x846ca68bu;
    v ^= v >> 16;
    return v;
}

DEVICE UINT ct_hash(UINT key, UINT stream, int a, int b) {
    UINT h = ct_mix(key ^ (stream * 0x9e3779b9u));
    h = ct_mix(h ^ ((UINT)a * 0x85ebca6bu));
    return ct_mix(h ^ ((UINT)b * 0xc2b2ae35u));
}

/* [0, 1) and [-1, 1), exact in float. */
DEVICE float ct_u01(UINT h) { return (float)(h >> 8) * (1.0f / 16777216.0f); }
DEVICE float ct_s11(UINT h) { return ct_u01(h) * 2.0f - 1.0f; }

/* Full-resolution row / column shown by a canvas row / column. */
DEVICE int ct_full_row(CT_PARAMS p, int row) { return (int)FFLOOR(((float)row + 0.5f) * p.inv_ds_y); }
DEVICE int ct_full_col(CT_PARAMS p, int col) { return (int)FFLOOR(((float)col + 0.5f) * p.inv_ds_x); }

/* Field of a full-resolution row: bands of size[0] even rows, then size[1] odd rows. */
DEVICE int ct_field(CT_PARAMS p, int fr) {
    const int period = p.size[0] + p.size[1];
    int m = fr % period;
    if (m < 0) m += period;
    return m < p.size[0] ? 0 : 1;
}

/* Horizontal displacement of a full-resolution row, in full-resolution pixels. */
DEVICE float ct_displacement(CT_PARAMS p, int f, int fr) {
    const float adj = p.hadj[f], infl = p.influence;
    const float w = (float)p.full_w, h = (float)p.full_h;
    const float t = (float)fr / h;
    const int method = p.method[f];
    if (method == 1) return -adj * (w / 100.0f) * (infl / 1000.0f);
    if (method == 2) return t * (-adj / 100.0f * w) * (infl / 100.0f);
    if (method == 3) {
        const float hi = (float)(int)(adj / 100.0f * w / 4.0f);
        return (float)(int)(ct_u01(ct_hash(p.key, CT_S_ROW, fr, 0)) * hi) * (infl / 1000.0f);
    }
    if (method == 4) {
        const int r = (int)(t * 4096.0f) + (int)(infl * 9.0f);
        int m = (int)(1.0f + infl / 1000.0f);
        if (m < 1) m = 1;
        return (float)(r % m - 10) * (-adj / (w / 10.0f));
    }
    if (method == 5) {
        const int r = (int)(t * 4096.0f);
        return FSIN(((float)r + infl) / (h / 40.0f)) * (-adj / 400.0f * w);
    }
    if (method == 6) {
        const int r = (int)(t * 4096.0f);
        return FSIN((float)r / ((h * infl + 0.0001f) / 100.0f)) * (-adj / 200.0f * w);
    }
    return 0.0f;
}

/* Canvas pixel (x, y), possibly outside the canvas, by edge mode; the input
 * buffer covers part of the canvas (transparent elsewhere). */
DEVICE float4 ct_fetch(GBUF_CONST in, CT_PARAMS p, int x, int y) {
    const int w = p.canvas_w, h = p.canvas_h;
    if (x < 0 || y < 0 || x >= w || y >= h) {
        if (p.edges == 1) return F4(0.0f, 0.0f, 0.0f, 0.0f);
        if (p.edges == 4) return F4(0.0f, 0.0f, 0.0f, 1.0f);
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

/* Unpremultiplied colour (x = B, y = G, z = R), alpha in w. */
DEVICE float4 ct_straight(float4 v) {
    if (v.w <= 0.0f) return F4(0.0f, 0.0f, 0.0f, 0.0f);
    const float k = 1.0f / v.w;
    return F4(v.x * k, v.y * k, v.z * k, v.w);
}

/* Share of the full-resolution columns under canvas column x that the comb keeps. */
DEVICE float ct_comb_pass(CT_PARAMS p, int comb, int x) {
    if (comb == 0) return 1.0f;
    const int c0 = (int)FFLOOR((float)x * p.inv_ds_x + 0.5f);
    int c1 = (int)FFLOOR((float)(x + 1) * p.inv_ds_x + 0.5f);
    if (c1 <= c0) c1 = c0 + 1;
    int kept = 0;
    for (int c = c0; c < c1; ++c)
        if ((c & comb) == 0) ++kept;
    return (float)kept / (float)(c1 - c0);
}

KERNEL(crosstalk_render, CrosstalkParams)
    if (x >= p.out_w || y >= p.out_h) return;
    const int cx = x + p.out_ox, cy = y + p.out_oy;
    const int fr = ct_full_row(p, cy), fc = ct_full_col(p, cx);
    const int f = ct_field(p, fr);
    const float4 orig = ct_fetch(a, p, cx, cy);

    /* Displaced source, whole pixels. */
    const float shift = ct_displacement(p, f, fr) * p.ds_x;
    const int sx = (int)FFLOOR((float)cx + shift + 0.5f);
    const float4 s = ct_straight(ct_fetch(a, p, sx, cy));

    /* Tint (luminance -> tint colour) mixed with the source by Process Input. */
    float tb = s.x, tg = s.y, tr = s.z;
    if (p.tint[f]) {
        const float l = (s.x + s.y + s.z) / 3.0f;
        tb = l * p.tint_b[f];
        tg = l * p.tint_g[f];
        tr = l * p.tint_r[f];
    }
    const float pr = p.process[f], keep = 1.0f - pr;
    float vb = keep * s.x + pr * tb, vg = keep * s.y + pr * tg, vr = keep * s.z + pr * tr;

    /* Noise: one value per pixel, or one per channel (Chroma Noise). */
    const float n = p.noise[f];
    const float nr = ct_s11(ct_hash(p.key, CT_S_NOISE, fc, fr)) * n;
    const float ng = p.chroma[f] ? ct_s11(ct_hash(p.key, CT_S_NOISE_G, fc, fr)) * n : nr;
    const float nb = p.chroma[f] ? ct_s11(ct_hash(p.key, CT_S_NOISE_B, fc, fr)) * n : nr;
    const float lum = p.lum[f];
    if (n > 128.0f / 255.0f) { /* very strong noise replaces the picture */
        vb = lum + 2.0f * nb;
        vg = lum + 2.0f * ng;
        vr = lum + 2.0f * nr;
    } else {
        vb += lum + nb;
        vg += lum + ng;
        vr += lum + nr;
    }

    /* Echo of the next pixels along the line (Decay), signal centred on 0.5. */
    const int decay = p.decay[f];
    if (decay > 1) {
        const float mb = keep * s.x + pr * tb, mg = keep * s.y + pr * tg, mr = keep * s.z + pr * tr;
        for (int k = 0; k < decay; ++k) {
            const int tx = (int)FFLOOR((float)cx + shift + (float)k * p.ds_x + 0.5f);
            const float4 e = ct_straight(ct_fetch(a, p, tx, cy));
            if (p.smear[f]) {
                const float wk = 1.0f / (float)(k + 2);
                vb += ((k == 0 ? mb : e.x) - 0.5f) * wk;
                vg += ((k == 0 ? mg : e.y) - 0.5f) * wk;
                vr += ((k == 0 ? mr : e.z) - 0.5f) * wk;
            } else {
                const float t = ct_s11(ct_hash(p.key, CT_S_TAP, fc, fr * 32 + k)) * n * 0.5f;
                const float wk = 1.0f - (float)k / (float)decay;
                vb += (e.x - 0.5f + t) * wk;
                vg += (e.y - 0.5f + t) * wk;
                vr += (e.z - 0.5f + t) * wk;
            }
        }
    }

    const float hi = p.color_max;
    vb = FMIN(FMAX(vb, 0.0f), hi);
    vg = FMIN(FMAX(vg, 0.0f), hi);
    vr = FMIN(FMAX(vr, 0.0f), hi);

    /* The source comes back on top, weighted by 1 - Process Input: dark areas
     * that the echo and Luminance pushed to black keep the picture. */
    vb = FMIN(vb + keep * s.x, hi);
    vg = FMIN(vg + keep * s.y, hi);
    vr = FMIN(vr + keep * s.z, hi);

    /* Comb filter: blanked columns turn transparent. */
    const float alpha = s.w * ct_comb_pass(p, p.comb[f], cx);
    float4 o = F4(vb * alpha, vg * alpha, vr * alpha, alpha);

    const float bl = p.blend, bk = 1.0f - bl;
    o = F4(o.x * bk + orig.x * bl, o.y * bk + orig.y * bl, o.z * bk + orig.z * bl, o.w * bk + orig.w * bl);
    dst[y * p.out_pitch + x] = o;
KERNEL_END
