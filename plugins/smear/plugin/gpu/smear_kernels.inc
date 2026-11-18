/* smear_kernels.inc — Smear's GPU passes, one source for CUDA, OpenCL and Metal.
 *
 * Mirrors core/include/smear_core.h (optical flow) and smear_render.h (blur)
 * pass for pass, with the same float arithmetic; tests/test_gpu_path.cpp runs
 * this file on the CPU (tests/cpu_prelude.h) and compares it with the CPU
 * renderer. Images: float4, x = B, y = G, z = R, w = A (premultiplied).
 * Flow fields: float2. Parameter blocks: smear_gpu_types.h.
 *
 * Each backend prepends a prelude defining the macros below. Tremor's preludes
 * already define DEVICE, PRIV, GBUF, GBUF_CONST, F4 and KERNEL_END; Smear adds:
 *
 *   GBUF2_CONST / GBUF2   read-only / writable float2 buffer pointer
 *   F2(a, b)              float2 constructor
 *   KERNEL_B2 … KERNEL_B6 (name, P, T0, a0, T1, a1, …): kernel `name` with N
 *       buffer arguments (type Tk, name ak, in order) followed by the parameter
 *       block `p` of type P, and `const int x, y` = global thread position.
 *       The types passed are GBUF/GBUF_CONST/GBUF2/GBUF2_CONST.
 *
 * Reference definitions (paste into the preludes):
 *
 *   OpenCL C:
 *     #define GBUF2_CONST __global const float2*
 *     #define GBUF2 __global float2*
 *     #define F2(a, b) ((float2)((a), (b)))
 *     #define SMEAR_XY const int x = (int)get_global_id(0); const int y = (int)get_global_id(1);
 *     #define KERNEL_B2(n, P, T0, a0, T1, a1) __kernel void n(T0 a0, T1 a1, P p) { SMEAR_XY
 *     #define KERNEL_B3(n, P, T0, a0, T1, a1, T2, a2) __kernel void n(T0 a0, T1 a1, T2 a2, P p) { SMEAR_XY
 *     … likewise up to KERNEL_B6.
 *
 *   Metal (buffer k at [[buffer(k)]], the parameter block right after the last buffer):
 *     #define GBUF2_CONST device const float2*
 *     #define GBUF2 device float2*
 *     #define F2(a, b) float2((a), (b))
 *     #define SMEAR_XY(P) uint2 gid [[thread_position_in_grid]]) { const P p = p_in; \
 *         const int x = int(gid.x); const int y = int(gid.y);
 *     #define KERNEL_B2(n, P, T0, a0, T1, a1) kernel void n(T0 a0 [[buffer(0)]], \
 *         T1 a1 [[buffer(1)]], constant P& p_in [[buffer(2)]], SMEAR_XY(P)
 *     … likewise up to KERNEL_B6 (parameter block at [[buffer(6)]]).
 *
 *   CUDA:
 *     #define GBUF2_CONST const float2*
 *     #define GBUF2 float2*
 *     #define F2(a, b) make_float2((a), (b))
 *     #define SMEAR_XY const int x = int(blockIdx.x * blockDim.x + threadIdx.x); \
 *                      const int y = int(blockIdx.y * blockDim.y + threadIdx.y);
 *     #define KERNEL_B2(n, P, T0, a0, T1, a1) extern "C" __global__ void n(T0 a0, T1 a1, P p) { SMEAR_XY
 *     … likewise up to KERNEL_B6.
 *
 * The code only uses float4 +, - and * float (already in Tremor's CUDA
 * prelude), no float2 operators, and these functions: fabs, floor, ceil, sqrt,
 * exp, atan2, fmin, fmax (floats) and min, max (ints). Kernels with a
 * read-only buffer they do not need (flagged by has_* = 0) are still given a
 * valid buffer by the host.
 */

/* --- Helpers ------------------------------------------------------------------ */

DEVICE int smear_clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
DEVICE float smear_clampf(float v, float lo, float hi) { return fmin(fmax(v, lo), hi); }
DEVICE float smear_smooth(float a, float b, float v) {
    const float t = smear_clampf((v - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
/* 8 neighbour directions: (-1,-1) (0,-1) (1,-1) (-1,0) (1,0) (-1,1) (0,1) (1,1). */
DEVICE int smear_offx(int k) { return (k == 0 || k == 3 || k == 5) ? -1 : ((k == 1 || k == 6) ? 0 : 1); }
DEVICE int smear_offy(int k) { return k < 3 ? -1 : (k < 5 ? 0 : 1); }
DEVICE int smear_origin(int i, int stride, int patch, int size) { return min(i * stride, size - patch); }

DEVICE float smear_gray(float4 c) { return 0.75f * (0.299f * c.z + 0.587f * c.y + 0.114f * c.x) + 0.25f * c.w; }

DEVICE float4 smear_fetch4(GBUF_CONST b, int pitch, int w, int h, int x, int y) {
    return b[smear_clampi(y, 0, h - 1) * pitch + smear_clampi(x, 0, w - 1)];
}

/* Bilinear float4 sample, clamped to the edges; pixel centres at integers. */
DEVICE float4 smear_sample4(GBUF_CONST b, int pitch, int w, int h, float x, float y) {
    x = smear_clampf(x, -1.0f, (float)w);
    y = smear_clampf(y, -1.0f, (float)h);
    const float fx0 = floor(x), fy0 = floor(y);
    const float fx = x - fx0, fy = y - fy0;
    const int ix = (int)fx0, iy = (int)fy0;
    const int x0 = smear_clampi(ix, 0, w - 1), x1 = smear_clampi(ix + 1, 0, w - 1);
    const int y0 = smear_clampi(iy, 0, h - 1), y1 = smear_clampi(iy + 1, 0, h - 1);
    const float4 p00 = b[y0 * pitch + x0], p10 = b[y0 * pitch + x1];
    const float4 p01 = b[y1 * pitch + x0], p11 = b[y1 * pitch + x1];
    const float4 top = p00 + (p10 - p00) * fx;
    const float4 bottom = p01 + (p11 - p01) * fx;
    return top + (bottom - top) * fy;
}

DEVICE float2 smear_sample2(GBUF2_CONST b, int pitch, int w, int h, float x, float y) {
    x = smear_clampf(x, -1.0f, (float)w);
    y = smear_clampf(y, -1.0f, (float)h);
    const float fx0 = floor(x), fy0 = floor(y);
    const float fx = x - fx0, fy = y - fy0;
    const int ix = (int)fx0, iy = (int)fy0;
    const int x0 = smear_clampi(ix, 0, w - 1), x1 = smear_clampi(ix + 1, 0, w - 1);
    const int y0 = smear_clampi(iy, 0, h - 1), y1 = smear_clampi(iy + 1, 0, h - 1);
    const float2 p00 = b[y0 * pitch + x0], p10 = b[y0 * pitch + x1];
    const float2 p01 = b[y1 * pitch + x0], p11 = b[y1 * pitch + x1];
    const float tx = p00.x + (p10.x - p00.x) * fx, bx = p01.x + (p11.x - p01.x) * fx;
    const float ty = p00.y + (p10.y - p00.y) * fx, by = p01.y + (p11.y - p01.y) * fx;
    return F2(tx + (bx - tx) * fy, ty + (by - ty) * fy);
}

/* Brightness of `to` (pyramid cell .y) at a fractional position. */
DEVICE float smear_sample_to(GBUF_CONST L, int pitch, int w, int h, float x, float y) {
    x = smear_clampf(x, -1.0f, (float)w);
    y = smear_clampf(y, -1.0f, (float)h);
    const float fx0 = floor(x), fy0 = floor(y);
    const float fx = x - fx0, fy = y - fy0;
    const int ix = (int)fx0, iy = (int)fy0;
    const int x0 = smear_clampi(ix, 0, w - 1), x1 = smear_clampi(ix + 1, 0, w - 1);
    const int y0 = smear_clampi(iy, 0, h - 1), y1 = smear_clampi(iy + 1, 0, h - 1);
    const float p00 = L[y0 * pitch + x0].y, p10 = L[y0 * pitch + x1].y;
    const float p01 = L[y1 * pitch + x0].y, p11 = L[y1 * pitch + x1].y;
    const float top = p00 + (p10 - p00) * fx, bottom = p01 + (p11 - p01) * fx;
    return top + (bottom - top) * fy;
}

/* --- Optical flow -------------------------------------------------------------- */

KERNEL_B3(smear_flow_gray, SmearGrayParams, GBUF_CONST, from, GBUF_CONST, to, GBUF, dst)
    if (x >= p.w || y >= p.h) return;
    const float l = smear_gray(smear_fetch4(from, p.from_pitch, p.w, p.h, x - 1, y));
    const float r = smear_gray(smear_fetch4(from, p.from_pitch, p.w, p.h, x + 1, y));
    const float u = smear_gray(smear_fetch4(from, p.from_pitch, p.w, p.h, x, y - 1));
    const float d = smear_gray(smear_fetch4(from, p.from_pitch, p.w, p.h, x, y + 1));
    dst[y * p.dst_pitch + x] = F4(smear_gray(smear_fetch4(from, p.from_pitch, p.w, p.h, x, y)),
                                  smear_gray(smear_fetch4(to, p.to_pitch, p.w, p.h, x, y)), 0.5f * (r - l), 0.5f * (d - u));
KERNEL_END

/* [1 3 3 1] / 8 tent of the finer level around coarse cell (x, y); .x and .y. */
DEVICE float2 smear_tent(GBUF_CONST src, int pitch, int sw, int sh, int x, int y) {
    float a = 0.0f, b = 0.0f;
    for (int j = 0; j < 4; ++j) {
        const int sy = smear_clampi(2 * y - 1 + j, 0, sh - 1);
        const float wy = (j == 0 || j == 3) ? 0.125f : 0.375f;
        for (int i = 0; i < 4; ++i) {
            const int sx = smear_clampi(2 * x - 1 + i, 0, sw - 1);
            const float k = ((i == 0 || i == 3) ? 0.125f : 0.375f) * wy;
            const float4 s = src[sy * pitch + sx];
            a += k * s.x;
            b += k * s.y;
        }
    }
    return F2(a, b);
}

KERNEL_B2(smear_flow_downsample, SmearDownParams, GBUF_CONST, src, GBUF, dst)
    if (x >= p.w || y >= p.h) return;
    const float2 c = smear_tent(src, p.src_pitch, p.sw, p.sh, x, y);
    const float l = smear_tent(src, p.src_pitch, p.sw, p.sh, smear_clampi(x - 1, 0, p.w - 1), y).x;
    const float r = smear_tent(src, p.src_pitch, p.sw, p.sh, smear_clampi(x + 1, 0, p.w - 1), y).x;
    const float u = smear_tent(src, p.src_pitch, p.sw, p.sh, x, smear_clampi(y - 1, 0, p.h - 1)).x;
    const float d = smear_tent(src, p.src_pitch, p.sw, p.sh, x, smear_clampi(y + 1, 0, p.h - 1)).x;
    dst[y * p.dst_pitch + x] = F4(c.x, c.y, 0.5f * (r - l), 0.5f * (d - u));
KERNEL_END

/* Sum of squared mean-normalised differences between the patch at (ox, oy)
 * and `to` displaced by (ux, uy). */
DEVICE float smear_patch_cost(GBUF_CONST L, int pitch, int w, int h, int ox, int oy, int ps, float mt, float var_t,
                              float ux, float uy) {
    float sw = 0.0f, sww = 0.0f, swt = 0.0f;
    for (int ky = 0; ky < ps; ++ky)
        for (int kx = 0; kx < ps; ++kx) {
            const float wv = smear_sample_to(L, pitch, w, h, (float)(ox + kx) + ux, (float)(oy + ky) + uy);
            sw += wv;
            sww += wv * wv;
            swt += wv * (L[(oy + ky) * pitch + ox + kx].x - mt);
        }
    return (sww - sw * sw / (float)(ps * ps)) - 2.0f * swt + var_t;
}

/* Integer displacements within ±r of (ax, ay) rounded; keeps the best. */
DEVICE void smear_integer_search(GBUF_CONST L, int lp, int w, int h, int ox, int oy, int ps, float mt, float var_t,
                                 float ax, float ay, int r, PRIV float* best_x, PRIV float* best_y, PRIV float* best_e) {
    const float inv_n = 1.0f / (float)(ps * ps);
    const int rx = (int)floor(ax + 0.5f), ry = (int)floor(ay + 0.5f);
    for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx) {
            float sw = 0.0f, sww = 0.0f, swt = 0.0f;
            for (int ky = 0; ky < ps; ++ky) {
                const int sy = smear_clampi(oy + ky + ry + dy, 0, h - 1);
                for (int kx = 0; kx < ps; ++kx) {
                    const int sx = smear_clampi(ox + kx + rx + dx, 0, w - 1);
                    const float wv = L[sy * lp + sx].y;
                    sw += wv;
                    sww += wv * wv;
                    swt += wv * (L[(oy + ky) * lp + ox + kx].x - mt);
                }
            }
            const float e = (sww - sw * sw * inv_n) - 2.0f * swt + var_t;
            if (e < *best_e) {
                *best_e = e;
                *best_x = (float)(rx + dx);
                *best_y = (float)(ry + dy);
            }
        }
}

/* One patch: candidates from the coarse flow, integer search (and a wide
 * rescue search if still poorly matched), then
 * inverse-compositional Gauss-Newton steps, keeping the best estimate seen. */
KERNEL_B3(smear_flow_patches, SmearPatchParams, GBUF_CONST, L, GBUF2_CONST, coarse, GBUF2, patches)
    if (x >= p.nx || y >= p.ny) return;
    const int ps = p.patch, n = ps * ps, lp = p.level_pitch;
    const int ox = smear_origin(x, p.stride, ps, p.w), oy = smear_origin(y, p.stride, ps, p.h);
    const float cx = (float)ox + (float)(ps - 1) * 0.5f, cy = (float)oy + (float)(ps - 1) * 0.5f;
    float ux = 0.0f, uy = 0.0f;
    if (p.has_coarse) {
        const float2 c = smear_sample2(coarse, p.coarse_pitch, p.cw, p.ch, cx * 0.5f - 0.25f, cy * 0.5f - 0.25f);
        ux = 2.0f * c.x;
        uy = 2.0f * c.y;
    }
    float st = 0.0f, sgx = 0.0f, sgy = 0.0f;
    for (int ky = 0; ky < ps; ++ky)
        for (int kx = 0; kx < ps; ++kx) {
            const float4 t = L[(oy + ky) * lp + ox + kx];
            st += t.x;
            sgx += t.z;
            sgy += t.w;
        }
    const float inv_n = 1.0f / (float)n;
    const float mt = st * inv_n, mgx = sgx * inv_n, mgy = sgy * inv_n;
    float hxx = 0.0f, hxy = 0.0f, hyy = 0.0f, bx0 = 0.0f, by0 = 0.0f, var_t = 0.0f;
    for (int ky = 0; ky < ps; ++ky)
        for (int kx = 0; kx < ps; ++kx) {
            const float4 t = L[(oy + ky) * lp + ox + kx];
            const float gx = t.z - mgx, gy = t.w - mgy, d = t.x - mt;
            hxx += gx * gx;
            hxy += gx * gy;
            hyy += gy * gy;
            bx0 += gx * d;
            by0 += gy * d;
            var_t += d * d;
        }
    const float det = hxx * hyy - hxy * hxy;
    if (!(det > p.min_det * (float)n * (float)n)) {
        patches[y * p.patch_pitch + x] = F2(ux, uy);
        return;
    }
    const float inv_det = 1.0f / det;

    if (p.has_coarse) {
        float ce = smear_patch_cost(L, lp, p.w, p.h, ox, oy, ps, mt, var_t, ux, uy);
        for (int k = 0; k < 8; ++k) {
            const float px = cx + (float)(smear_offx(k) * ps), py = cy + (float)(smear_offy(k) * ps);
            const float2 c = smear_sample2(coarse, p.coarse_pitch, p.cw, p.ch, px * 0.5f - 0.25f, py * 0.5f - 0.25f);
            const float vx = 2.0f * c.x, vy = 2.0f * c.y;
            const float e = smear_patch_cost(L, lp, p.w, p.h, ox, oy, ps, mt, var_t, vx, vy);
            if (e < ce) {
                ce = e;
                ux = vx;
                uy = vy;
            }
        }
    }

    float best_x = ux, best_y = uy, best_e = 3.0e38f;
    if (p.radius > 0) {
        smear_integer_search(L, lp, p.w, p.h, ox, oy, ps, mt, var_t, ux, uy, p.radius, &best_x, &best_y, &best_e);
        ux = best_x;
        uy = best_y;
    }
    if (p.rescue > 0 && best_e > p.rescue_ratio * var_t && var_t > p.rescue_min_var * (float)n) {
        smear_integer_search(L, lp, p.w, p.h, ox, oy, ps, mt, var_t, ux, uy, p.rescue, &best_x, &best_y, &best_e);
        ux = best_x;
        uy = best_y;
    }
    for (int it = 0; it <= p.iterations; ++it) {
        float sw = 0.0f, sww = 0.0f, swt = 0.0f, bx = 0.0f, by = 0.0f;
        for (int ky = 0; ky < ps; ++ky)
            for (int kx = 0; kx < ps; ++kx) {
                const float4 t = L[(oy + ky) * lp + ox + kx];
                const float wv = smear_sample_to(L, lp, p.w, p.h, (float)(ox + kx) + ux, (float)(oy + ky) + uy);
                sw += wv;
                sww += wv * wv;
                swt += wv * (t.x - mt);
                bx += (t.z - mgx) * wv;
                by += (t.w - mgy) * wv;
            }
        const float e = (sww - sw * sw * inv_n) - 2.0f * swt + var_t;
        if (e < best_e) {
            best_e = e;
            best_x = ux;
            best_y = uy;
        }
        if (it == p.iterations) break;
        bx -= bx0;
        by -= by0;
        float dx = (hyy * bx - hxy * by) * inv_det, dy = (hxx * by - hxy * bx) * inv_det;
        const float len = sqrt(dx * dx + dy * dy);
        if (len > p.max_step) {
            dx *= p.max_step / len;
            dy *= p.max_step / len;
        }
        ux -= dx;
        uy -= dy;
    }
    patches[y * p.patch_pitch + x] = F2(best_x, best_y);
KERNEL_END

/* Inverse photometric error of flow (ux, uy) at level pixel (x, y). */
DEVICE float smear_photo_weight(GBUF_CONST L, int pitch, int w, int h, int x, int y, float ux, float uy, float eps) {
    const float e = fabs(smear_sample_to(L, pitch, w, h, (float)x + ux, (float)y + uy) - L[y * pitch + x].x);
    return 1.0f / fmax(e, eps);
}

KERNEL_B3(smear_flow_densify, SmearDensifyParams, GBUF_CONST, L, GBUF2_CONST, patches, GBUF2, flow)
    if (x >= p.w || y >= p.h) return;
    const int ps = p.patch, st = p.stride;
    const int i0 = max(0, (x - ps + 1 + st - 1) / st), j0 = max(0, (y - ps + 1 + st - 1) / st);
    float sx = 0.0f, sy = 0.0f, sw = 0.0f;
    for (int j = j0; j < p.ny; ++j) {
        const int oy = smear_origin(j, st, ps, p.h);
        if (oy > y) break;
        if (y >= oy + ps) continue;
        for (int i = i0; i < p.nx; ++i) {
            const int ox = smear_origin(i, st, ps, p.w);
            if (ox > x) break;
            if (x >= ox + ps) continue;
            const float2 u = patches[j * p.patch_pitch + i];
            const float wgt = smear_photo_weight(L, p.level_pitch, p.w, p.h, x, y, u.x, u.y, p.photo_eps);
            sx += wgt * u.x;
            sy += wgt * u.y;
            sw += wgt;
        }
    }
    flow[y * p.flow_pitch + x] = sw > 0.0f ? F2(sx / sw, sy / sw) : F2(0.0f, 0.0f);
KERNEL_END

KERNEL_B3(smear_flow_upsample, SmearUpsampleParams, GBUF_CONST, L, GBUF2_CONST, coarse, GBUF2, flow)
    if (x >= p.w || y >= p.h) return;
    const float sc = (float)p.scale, inv = 1.0f / sc;
    float qx = ((float)x + 0.5f) * inv - 0.5f, qy = ((float)y + 0.5f) * inv - 0.5f;
    qx = smear_clampf(qx, -1.0f, (float)p.cw);
    qy = smear_clampf(qy, -1.0f, (float)p.ch);
    const float fx0 = floor(qx), fy0 = floor(qy);
    const float fx = qx - fx0, fy = qy - fy0;
    const int ix = (int)fx0, iy = (int)fy0;
    float sx = 0.0f, sy = 0.0f, sw = 0.0f;
    for (int j = 0; j < 2; ++j)
        for (int i = 0; i < 2; ++i) {
            const int cxi = smear_clampi(ix + i, 0, p.cw - 1), cyi = smear_clampi(iy + j, 0, p.ch - 1);
            const float2 c = coarse[cyi * p.coarse_pitch + cxi];
            const float ux = c.x * sc, uy = c.y * sc;
            const float wb = (i == 0 ? 1.0f - fx : fx) * (j == 0 ? 1.0f - fy : fy);
            const float wgt = (wb + 1e-3f) * smear_photo_weight(L, p.level_pitch, p.w, p.h, x, y, ux, uy, p.photo_eps);
            sx += wgt * ux;
            sy += wgt * uy;
            sw += wgt;
        }
    flow[y * p.flow_pitch + x] = F2(sx / sw, sy / sw);
KERNEL_END

DEVICE float smear_window_error(GBUF_CONST L, int pitch, int w, int h, int x, int y, float ux, float uy) {
    float e = 0.0f;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const int px = smear_clampi(x + dx, 0, w - 1), py = smear_clampi(y + dy, 0, h - 1);
            e += fabs(smear_sample_to(L, pitch, w, h, (float)px + ux, (float)py + uy) - L[py * pitch + px].x);
        }
    return e;
}

KERNEL_B3(smear_flow_refine, SmearRefineParams, GBUF_CONST, L, GBUF2_CONST, src, GBUF2, dst)
    if (x >= p.w || y >= p.h) return;
    float2 best = src[y * p.src_pitch + x];
    float best_e = smear_window_error(L, p.level_pitch, p.w, p.h, x, y, best.x, best.y);
    for (int k = 0; k < 8; ++k) {
        const int px = smear_clampi(x + smear_offx(k) * p.dist, 0, p.w - 1);
        const int py = smear_clampi(y + smear_offy(k) * p.dist, 0, p.h - 1);
        const float2 u = src[py * p.src_pitch + px];
        if (fabs(u.x - best.x) + fabs(u.y - best.y) < 0.05f) continue;
        const float e = smear_window_error(L, p.level_pitch, p.w, p.h, x, y, u.x, u.y);
        if (e < best_e) {
            best_e = e;
            best = u;
        }
    }
    dst[y * p.dst_pitch + x] = best;
KERNEL_END

/* --- Motion blur ------------------------------------------------------------------ */

DEVICE int smear_inside(int w, int h, float px, float py) {
    return px > -0.5f && py > -0.5f && px < (float)w - 0.5f && py < (float)h - 0.5f;
}

/* Colour error between c and image b at (px, py); 1 outside the image. */
DEVICE float smear_match(float4 c, GBUF_CONST b, int pitch, int w, int h, float px, float py) {
    if (!smear_inside(w, h, px, py)) return 1.0f;
    const float4 s = smear_sample4(b, pitch, w, h, px, py);
    return 0.25f * (fabs(s.z - c.z) + fabs(s.y - c.y) + fabs(s.x - c.x) + fabs(s.w - c.w));
}

KERNEL_B6(smear_motion, SmearMotionParams, GBUF_CONST, cur, GBUF_CONST, prev, GBUF_CONST, next, GBUF2_CONST, to_prev,
          GBUF2_CONST, to_next, GBUF, mot)
    if (x >= p.w || y >= p.h) return;
    const float4 c = cur[y * p.cur_pitch + x];
    float vx = 0.0f, vy = 0.0f, occ = 0.0f;
    const float fxp = (float)x, fyp = (float)y;
    if (p.has_prev && p.has_next) {
        const float2 un = to_next[y * p.to_next_pitch + x], up = to_prev[y * p.to_prev_pitch + x];
        const float en = smear_match(c, next, p.next_pitch, p.w, p.h, fxp + un.x, fyp + un.y);
        const float ep = smear_match(c, prev, p.prev_pitch, p.w, p.h, fxp + up.x, fyp + up.y);
        const float wn = 1.0f - smear_smooth(p.photo_lo, p.photo_hi, en);
        const float wp = 1.0f - smear_smooth(p.photo_lo, p.photo_hi, ep);
        const float dx = un.x + up.x, dy = un.y + up.y;
        const float d = sqrt(dx * dx + dy * dy);
        const float speed = 0.5f * (sqrt(un.x * un.x + un.y * un.y) + sqrt(up.x * up.x + up.y * up.y));
        const float lo = p.cons_abs + p.cons_rel * speed;
        const float incons = smear_smooth(lo, 2.0f * lo, d);
        const float kn = (1.0f - incons) * 0.5f + incons * (wn + 1e-3f) / (wn + wp + 2e-3f);
        vx = kn * un.x - (1.0f - kn) * up.x;
        vy = kn * un.y - (1.0f - kn) * up.y;
        occ = (smear_inside(p.w, p.h, fxp + un.x, fyp + un.y) && smear_inside(p.w, p.h, fxp + up.x, fyp + up.y))
                  ? fabs(wn - wp) : 0.0f;
    } else if (p.has_next) {
        const float2 un = to_next[y * p.to_next_pitch + x];
        vx = un.x;
        vy = un.y;
        occ = smear_inside(p.w, p.h, fxp + un.x, fyp + un.y)
                  ? smear_smooth(p.photo_lo, p.photo_hi, smear_match(c, next, p.next_pitch, p.w, p.h, fxp + un.x, fyp + un.y))
                  : 0.0f;
    } else if (p.has_prev) {
        const float2 up = to_prev[y * p.to_prev_pitch + x];
        vx = -up.x;
        vy = -up.y;
        occ = smear_inside(p.w, p.h, fxp + up.x, fyp + up.y)
                  ? smear_smooth(p.photo_lo, p.photo_hi, smear_match(c, prev, p.prev_pitch, p.w, p.h, fxp + up.x, fyp + up.y))
                  : 0.0f;
    }
    float hx = vx * p.half_shutter, hy = vy * p.half_shutter;
    const float len = sqrt(hx * hx + hy * hy);
    if (len > p.max_radius) {
        hx *= p.max_radius / len;
        hy *= p.max_radius / len;
    }
    mot[y * p.mot_pitch + x] = F4(hx, hy, occ, 0.0f);
KERNEL_END

/* How well a centred velocity (vx, vy) explains pixel c in both neighbours. */
DEVICE float smear_motion_cost(float4 c, GBUF_CONST prev, GBUF_CONST next, PRIV const SmearMotionRefineParams* p,
                               int x, int y, float vx, float vy, PRIV float* occ) {
    float en = 0.0f, ep = 0.0f;
    int in_n = 1, in_p = 1;
    const float fx = (float)x, fy = (float)y;
    if (p->has_next) {
        in_n = smear_inside(p->w, p->h, fx + vx, fy + vy);
        en = smear_match(c, next, p->next_pitch, p->w, p->h, fx + vx, fy + vy);
    }
    if (p->has_prev) {
        in_p = smear_inside(p->w, p->h, fx - vx, fy - vy);
        ep = smear_match(c, prev, p->prev_pitch, p->w, p->h, fx - vx, fy - vy);
    }
    if (p->has_next && p->has_prev) {
        const float wn = 1.0f - smear_smooth(p->photo_lo, p->photo_hi, en);
        const float wp = 1.0f - smear_smooth(p->photo_lo, p->photo_hi, ep);
        *occ = (in_n && in_p) ? fabs(wn - wp) : 0.0f;
        return fmin(en, ep) + fmin(fmax(en, ep), p->photo_hi);
    }
    const float e = p->has_next ? en : ep;
    *occ = (p->has_next ? in_n : in_p) ? smear_smooth(p->photo_lo, p->photo_hi, e) : 0.0f;
    return e;
}

KERNEL_B5(smear_motion_refine, SmearMotionRefineParams, GBUF_CONST, cur, GBUF_CONST, prev, GBUF_CONST, next,
          GBUF_CONST, src, GBUF, dst)
    if (x >= p.w || y >= p.h) return;
    const float4 c = cur[y * p.cur_pitch + x];
    const float4 m0 = src[y * p.src_pitch + x];
    float occ0 = 0.0f;
    const float e0 = smear_motion_cost(c, prev, next, &p, x, y, m0.x * p.inv_half, m0.y * p.inv_half, &occ0);
    float4 best = m0;
    float best_e = e0 - p.margin;
    for (int k = 0; k < 8; ++k) {
        const int px = smear_clampi(x + smear_offx(k) * p.dist, 0, p.w - 1);
        const int py = smear_clampi(y + smear_offy(k) * p.dist, 0, p.h - 1);
        const float4 m = src[py * p.src_pitch + px];
        if (fabs(m.x - best.x) + fabs(m.y - best.y) < 0.01f) continue;
        float occ = 0.0f;
        const float e = smear_motion_cost(c, prev, next, &p, x, y, m.x * p.inv_half, m.y * p.inv_half, &occ);
        if (e < best_e) {
            best_e = e;
            best = F4(m.x, m.y, occ, 0.0f);
        }
    }
    dst[y * p.dst_pitch + x] = best;
KERNEL_END

KERNEL_B4(smear_tiles, SmearTileParams, GBUF_CONST, mot, GBUF_CONST, prior, GBUF2, tmax, GBUF, tocc)
    if (x >= p.tw || y >= p.th) return;
    const int x0 = x * p.tile, y0 = y * p.tile;
    const int x1 = min(x0 + p.tile, p.w), y1 = min(y0 + p.tile, p.h);
    float px = 0.0f, py = 0.0f, inv_s2 = 0.0f;
    if (p.has_prior) {
        const float4 pr = prior[y * p.prior_pitch + x];
        const float k = 1.0f / fmax(pr.z, 1e-6f);
        px = pr.x * k;
        py = pr.y * k;
        const float sg = p.sigma_abs + p.sigma_rel * sqrt(px * px + py * py);
        inv_s2 = 1.0f / (sg * sg);
    }
    float best = -1.0f, sx = 0.0f, sy = 0.0f, so = 0.0f, mx = 0.0f, my = 0.0f;
    for (int yy = y0; yy < y1; ++yy)
        for (int xx = x0; xx < x1; ++xx) {
            const float4 m = mot[yy * p.mot_pitch + xx];
            const float l2 = m.x * m.x + m.y * m.y;
            if (l2 > best) {
                best = l2;
                mx = m.x;
                my = m.y;
            }
            float wgt = m.z;
            if (p.has_prior) {
                const float dx = m.x - px, dy = m.y - py;
                wgt = wgt / (1.0f + (dx * dx + dy * dy) * inv_s2);
            }
            sx += wgt * m.x;
            sy += wgt * m.y;
            so += wgt;
        }
    tmax[y * p.tmax_pitch + x] = F2(mx, my);
    tocc[y * p.tocc_pitch + x] = F4(sx, sy, so, 0.0f);
KERNEL_END

KERNEL_B4(smear_neighbours, SmearNeighbourParams, GBUF2_CONST, tmax, GBUF_CONST, tocc, GBUF2, nmax, GBUF, nocc)
    if (x >= p.tw || y >= p.th) return;
    float best = -1.0f, nx = 0.0f, ny = 0.0f;
    for (int dy = -p.reach; dy <= p.reach; ++dy) {
        const int ty = y + dy;
        if (ty < 0 || ty >= p.th) continue;
        for (int dx = -p.reach; dx <= p.reach; ++dx) {
            const int tx = x + dx;
            if (tx < 0 || tx >= p.tw) continue;
            const float2 v = tmax[ty * p.tmax_pitch + tx];
            const float l2 = v.x * v.x + v.y * v.y;
            if (l2 <= best) continue;
            const int adx = dx < 0 ? -dx : dx, ady = dy < 0 ? -dy : dy;
            const int ring = max(adx, ady);
            if (ring > 1) {
                const float gap = (float)(ring - 1) * (float)p.tile;
                const float len = sqrt(l2);
                if (len <= gap) continue;
                const float ddx = (float)(-dx), ddy = (float)(-dy);
                const float cosang = fabs(v.x * ddx + v.y * ddy) / (len * sqrt(ddx * ddx + ddy * ddy));
                if (cosang < 0.5f) continue;
            }
            best = l2;
            nx = v.x;
            ny = v.y;
        }
    }
    const float lmax = sqrt(fmax(best, 0.0f));
    const int r = min(p.reach, p.bg_tiles + (int)ceil(p.bg_scale * lmax / (float)p.tile));
    float sx = 0.0f, sy = 0.0f, so = 0.0f;
    for (int ty = max(0, y - r); ty <= min(p.th - 1, y + r); ++ty)
        for (int tx = max(0, x - r); tx <= min(p.tw - 1, x + r); ++tx) {
            const float4 o = tocc[ty * p.tocc_pitch + tx];
            sx += o.x;
            sy += o.y;
            so += o.z;
        }
    nmax[y * p.nmax_pitch + x] = F2(nx, ny);
    nocc[y * p.nocc_pitch + x] = F4(sx, sy, so, 0.0f);
KERNEL_END

/* Front score of a pixel with half-streak (hx, hy) against the local background. */
DEVICE float smear_front(float hx, float hy, float bx, float by, float gate, PRIV const SmearGatherParams* p) {
    const float dx = (hx - bx) * p->to_velocity, dy = (hy - by) * p->to_velocity;
    const float dv = sqrt(dx * dx + dy * dy);
    const float bs = sqrt(bx * bx + by * by) * p->to_velocity;
    const float lo = p->layer_abs + p->layer_rel * bs;
    return gate * smear_smooth(lo, 2.0f * lo + 1.0f, dv);
}

DEVICE float smear_cover(float d, float len) { return smear_clampf(len + 0.5f - d, 0.0f, 1.0f); }

/* Vector view: direction = hue, speed = saturation, front = brightness, occlusion = magenta. */
DEVICE float4 smear_vector_colour(float hx, float hy, float to_velocity, float front, float occ) {
    const float vx = hx * to_velocity, vy = hy * to_velocity;
    const float speed = sqrt(vx * vx + vy * vy);
    const float hue = (atan2(vy, vx) * 0.15915494f + 0.5f) * 6.0f;
    const float r = smear_clampf(fabs(hue - 3.0f) - 1.0f, 0.0f, 1.0f);
    const float g = smear_clampf(2.0f - fabs(hue - 2.0f), 0.0f, 1.0f);
    const float b = smear_clampf(2.0f - fabs(hue - 4.0f), 0.0f, 1.0f);
    const float sat = 1.0f - exp(-speed * 0.0625f);
    const float val = 0.5f + 0.5f * front;
    float pr = val * (1.0f - sat + sat * r), pg = val * (1.0f - sat + sat * g), pb = val * (1.0f - sat + sat * b);
    const float o = 0.6f * occ;
    pr += (1.0f - pr) * o;
    pg *= 1.0f - o;
    pb += (1.0f - pb) * o;
    return F4(pb, pg, pr, 1.0f);
}

KERNEL_B5(smear_gather, SmearGatherParams, GBUF_CONST, cur, GBUF_CONST, mot, GBUF2_CONST, nmax, GBUF_CONST, nocc,
          GBUF, dst)
    if (x >= p.out_w || y >= p.out_h) return;
    const int sx = x + p.out_ox, sy = y + p.out_oy;
    if (sx < 0 || sy < 0 || sx >= p.w || sy >= p.h) {
        dst[y * p.out_pitch + x] = F4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }
    const float4 c = cur[sy * p.cur_pitch + sx];
    const float4 m = mot[sy * p.mot_pitch + sx];
    const int tx = sx / p.tile, ty = sy / p.tile;

    /* Local background, interpolated between tile centres. */
    float qx = ((float)sx + 0.5f) / (float)p.tile - 0.5f, qy = ((float)sy + 0.5f) / (float)p.tile - 0.5f;
    qx = smear_clampf(qx, -1.0f, (float)p.tw);
    qy = smear_clampf(qy, -1.0f, (float)p.th);
    const float fqx = floor(qx), fqy = floor(qy);
    const float bfx = qx - fqx, bfy = qy - fqy;
    const int bix = (int)fqx, biy = (int)fqy;
    const int bx0 = smear_clampi(bix, 0, p.tw - 1), bx1 = smear_clampi(bix + 1, 0, p.tw - 1);
    const int by0 = smear_clampi(biy, 0, p.th - 1), by1 = smear_clampi(biy + 1, 0, p.th - 1);
    const float4 o00 = nocc[by0 * p.nocc_pitch + bx0], o10 = nocc[by0 * p.nocc_pitch + bx1];
    const float4 o01 = nocc[by1 * p.nocc_pitch + bx0], o11 = nocc[by1 * p.nocc_pitch + bx1];
    const float4 otop = o00 + (o10 - o00) * bfx;
    const float4 obot = o01 + (o11 - o01) * bfx;
    const float4 osum = otop + (obot - otop) * bfy;
    const float inv_mass = 1.0f / fmax(osum.z, 1e-6f);
    const float bgx = osum.x * inv_mass, bgy = osum.y * inv_mass;
    const float gate = smear_smooth(p.mass_lo, p.mass_hi, osum.z);
    const float zx = smear_front(m.x, m.y, bgx, bgy, gate, &p);

    if (p.show_vectors) {
        dst[y * p.out_pitch + x] = smear_vector_colour(m.x, m.y, p.to_velocity, zx, m.z);
        return;
    }

    const float2 nm = nmax[ty * p.nmax_pitch + tx];
    const float ln = sqrt(nm.x * nm.x + nm.y * nm.y);
    const float lx = sqrt(m.x * m.x + m.y * m.y);
    if (ln < 0.5f) {
        dst[y * p.out_pitch + x] = c;
        return;
    }
    /* Own vector dominant: nothing nearby moves faster over us, all samples on our streak. */
    const int budget = max(4, p.max_samples);
    const float dnx = nm.x - m.x, dny = nm.y - m.y;
    const int own_dominant = dnx * dnx + dny * dny < 0.25f;
    int nb = 0;
    if (lx >= 0.5f) nb = min(own_dominant ? budget : budget / 2, max(2, 2 * (int)ceil(lx)));
    nb = (nb / 2) * 2;
    int na = 0;
    if (!own_dominant) {
        na = min(budget - nb, max(2, 2 * (int)ceil(ln)));
        na = max(2, (na / 2) * 2);
    }

    const float ndx = nm.x / ln, ndy = nm.y / ln;
    const float lxn = fabs(m.x * ndx + m.y * ndy);
    const float fpx = (float)sx, fpy = (float)sy;

    /* Front coverage: layers whose streaks pass over this pixel. */
    float fw = 0.0f;
    float4 fc = F4(0.0f, 0.0f, 0.0f, 0.0f);
    const float da = na > 0 ? 2.0f * ln / (float)na : 0.0f;
    for (int i = 0; i < na; ++i) {
        const float s = -1.0f + (2.0f * (float)i + 1.0f) / (float)na;
        const float px = fpx + s * nm.x, py = fpy + s * nm.y; /* outside: edge pixels extend */
        const int ix = smear_clampi((int)floor(px + 0.5f), 0, p.w - 1), iy = smear_clampi((int)floor(py + 0.5f), 0, p.h - 1);
        const float4 my = mot[iy * p.mot_pitch + ix];
        const float zy = smear_front(my.x, my.y, bgx, bgy, gate, &p);
        const float front = smear_clampf((zy - zx) * 2.0f, 0.0f, 1.0f);
        const float back = smear_clampf((zx - zy) * 2.0f, 0.0f, 1.0f);
        const float same = 1.0f - front - back;
        const float ly = fabs(my.x * ndx + my.y * ndy);
        const float excess = smear_clampf((ly - lxn) / (ly + 1.0f), 0.0f, 1.0f);
        const float d = fabs(s) * ln;
        const float wgt = smear_cover(d, ly) / (2.0f * ly + 1.0f) * da * (front + same * excess);
        if (wgt <= 0.0f) continue;
        fw += wgt;
        fc = fc + smear_sample4(cur, p.cur_pitch, p.w, p.h, px, py) * wgt;
    }

    /* Own layer (and whatever is behind it) blurred along our own streak. */
    const float wc = (nb > 0 ? 0.01f : 1.0f) / (2.0f * lx + 1.0f);
    float bw = wc;
    float4 bc = c * wc;
    if (nb > 0) {
        const float db = 2.0f * lx / (float)nb;
        for (int i = 0; i < nb; ++i) {
            const float s = -1.0f + (2.0f * (float)i + 1.0f) / (float)nb;
            const float px = fpx + s * m.x, py = fpy + s * m.y;
            const int ix = smear_clampi((int)floor(px + 0.5f), 0, p.w - 1), iy = smear_clampi((int)floor(py + 0.5f), 0, p.h - 1);
            const float4 my = mot[iy * p.mot_pitch + ix];
            const float zy = smear_front(my.x, my.y, bgx, bgy, gate, &p);
            const float front = smear_clampf((zy - zx) * 2.0f, 0.0f, 1.0f);
            const float d = fabs(s) * lx;
            const float wgt = smear_cover(d, lx) / (2.0f * lx + 1.0f) * db * (1.0f - front);
            if (wgt <= 0.0f) continue;
            bw += wgt;
            bc = bc + smear_sample4(cur, p.cur_pitch, p.w, p.h, px, py) * wgt;
        }
    }
    float4 o = bc * (1.0f / bw);
    if (fw > 0.0f) {
        const float alpha = smear_clampf(fw, 0.0f, 1.0f);
        o = fc * (alpha / fw) + o * (1.0f - alpha);
    }
    dst[y * p.out_pitch + x] = o;
KERNEL_END

KERNEL_B2(smear_copy, SmearCopyParams, GBUF_CONST, src, GBUF, dst)
    if (x >= p.out_w || y >= p.out_h) return;
    const int sx = x + p.out_ox, sy = y + p.out_oy;
    dst[y * p.out_pitch + x] = (sx < 0 || sy < 0 || sx >= p.w || sy >= p.h) ? F4(0.0f, 0.0f, 0.0f, 0.0f)
                                                                            : src[sy * p.src_pitch + sx];
KERNEL_END
