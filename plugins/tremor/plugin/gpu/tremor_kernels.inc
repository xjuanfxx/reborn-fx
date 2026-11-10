/* tremor_kernels.inc — Tremor's GPU passes, one source for CUDA, OpenCL and Metal.
 *
 * Each backend prepends a prelude defining DEVICE, PRIV, GBUF, GBUF_CONST, F4,
 * KERNEL1/KERNEL2 and KERNEL_END. Mirrors core/include/tremor_render.h.
 * Pixels: x = B, y = G, z = R, w = A (premultiplied).
 */

/* Pixel (x, y), possibly outside the image, by edge mode (tremor::Edges). */
DEVICE float4 tremor_fetch(GBUF_CONST buf, int pitch, int w, int h, int x, int y, int edges) {
    if (x < 0 || y < 0 || x >= w || y >= h) {
        if (edges == 1) return F4(0.0f, 0.0f, 0.0f, 0.0f);
        if (edges == 4) return F4(0.0f, 0.0f, 0.0f, 1.0f);
        if (edges == 3) {
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

/* Bilinear sample; pixel centers at integer coordinates. */
DEVICE float4 tremor_sample(GBUF_CONST buf, int pitch, int w, int h, float x, float y, int edges) {
    if (!(fabs(x) < 1e8f && fabs(y) < 1e8f))
        return edges == 4 ? F4(0.0f, 0.0f, 0.0f, 1.0f) : F4(0.0f, 0.0f, 0.0f, 0.0f);
    const float fx0 = floor(x), fy0 = floor(y);
    const int x0 = (int)fx0, y0 = (int)fy0;
    const float fx = x - fx0, fy = y - fy0;
    const float4 p00 = tremor_fetch(buf, pitch, w, h, x0, y0, edges);
    const float4 p10 = tremor_fetch(buf, pitch, w, h, x0 + 1, y0, edges);
    const float4 p01 = tremor_fetch(buf, pitch, w, h, x0, y0 + 1, edges);
    const float4 p11 = tremor_fetch(buf, pitch, w, h, x0 + 1, y0 + 1, edges);
    const float4 top = p00 + (p10 - p00) * fx;
    const float4 bottom = p01 + (p11 - p01) * fx;
    return top + (bottom - top) * fy;
}

DEVICE float4 tremor_input(GBUF_CONST src, PRIV const TremorSourceParams* p, int x, int y) {
    const int ix = x - p->in_ox, iy = y - p->in_oy;
    if (ix < 0 || iy < 0 || ix >= p->in_w || iy >= p->in_h) return F4(0.0f, 0.0f, 0.0f, 0.0f);
    return src[iy * p->in_pitch + ix];
}

KERNEL1(tremor_source, TremorSourceParams)
    if (x >= p.w || y >= p.h) return;
    float4 m = tremor_input(src, &p, x, y);
    int rebuild = 0; /* 1: average of the rows above and below, 2: [1 2 1] / 4 */
    if (y > 0 && y < p.h - 1) {
        if (p.mode == 2 && y % 2 == 0) rebuild = 1;
        if (p.mode == 3 && y % 2 == 1) rebuild = 1;
        if (p.mode == 4) rebuild = 2;
    }
    if (rebuild) {
        const float4 a = tremor_input(src, &p, x, y - 1), b = tremor_input(src, &p, x, y + 1);
        const float wa = rebuild == 1 ? 0.5f : 0.25f, wm = rebuild == 1 ? 0.0f : 0.5f;
        m = F4((a.x + b.x) * wa + m.x * wm, (a.y + b.y) * wa + m.y * wm, (a.z + b.z) * wa + m.z * wm, m.w);
    }
    dst[y * p.dst_pitch + x] = m;
KERNEL_END

KERNEL1(tremor_rotate, TremorRotateParams)
    if (x >= p.w || y >= p.h) return;
    const float dx = (float)x - p.cx, dy = (float)y - p.cy;
    const float4 r = tremor_sample(src, p.src_pitch, p.w, p.h, p.cx + dx * p.c0 - dy * p.s0, p.cy + dx * p.s0 + dy * p.c0, p.edges);
    const float4 g = tremor_sample(src, p.src_pitch, p.w, p.h, p.cx + dx * p.c1 - dy * p.s1, p.cy + dx * p.s1 + dy * p.c1, p.edges);
    const float4 b = tremor_sample(src, p.src_pitch, p.w, p.h, p.cx + dx * p.c2 - dy * p.s2, p.cy + dx * p.s2 + dy * p.c2, p.edges);
    dst[y * p.dst_pitch + x] = F4(b.x, g.y, r.z, (r.w + g.w + b.w) / 3.0f);
KERNEL_END

/* src = rotated canvas, src2 = filtered source canvas. */
KERNEL2(tremor_finish, TremorFinishParams)
    if (x >= p.out_w || y >= p.out_h) return;
    const int sx = x + p.ox, sy = y + p.oy;
    float4 s0 = F4(0.0f, 0.0f, 0.0f, 0.0f);
    if (sx >= 0 && sy >= 0 && sx < p.w && sy < p.h) s0 = src2[sy * p.src_pitch + sx];
    const float keep = 1.0f - p.blend;
    if (keep <= 0.0f) {
        dst[y * p.out_pitch + x] = s0;
        return;
    }
    float4 o = p.edges == 4 ? F4(0.0f, 0.0f, 0.0f, 1.0f) : F4(0.0f, 0.0f, 0.0f, 0.0f);
    if (p.zoom_valid)
        o = tremor_sample(src, p.rot_pitch, p.w, p.h, p.cx + ((float)sx - p.cx) * p.inv_zoom - p.shift_x,
                          p.cy + ((float)sy - p.cy) * p.inv_zoom - p.shift_y, p.edges);
    const float hi = p.color_max;
    o = F4(clamp(o.x + p.adj_b * o.w, 0.0f, hi), clamp(o.y + p.adj_g * o.w, 0.0f, hi),
           clamp(o.z + p.adj_r * o.w, 0.0f, hi), o.w);
    if (p.composite) o = o + s0 * (1.0f - o.w);
    dst[y * p.out_pitch + x] = o * keep + s0 * p.blend;
KERNEL_END
