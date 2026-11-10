// smear_core.h — Smear's image types and dense optical flow (CPU reference).
//
// Independent of After Effects. Everything is a pure function of its inputs:
// no global state, no caches, so frames render in any order and in parallel.
//
// Optical flow: coarse-to-fine patch inverse search with densification, after
// Kroeger et al. 2016, "Fast Optical Flow using Dense Inverse Search" (DIS),
// plus a few additions for large motions and sharp motion boundaries.
// Per pyramid level, from coarse to fine:
//
//   1. patch search: overlapping square patches of the `from` image. Each one
//      starts from the coarser level's flow, tried at its centre and one patch
//      away in eight directions (the side of a motion boundary the patch
//      mostly covers wins); then an integer search within ±radius (wide on the
//      two coarsest levels); then inverse-compositional Gauss-Newton steps
//      (the template Hessian is constant, so a step costs one warp of the
//      patch). Patches are mean-normalised (robust to small exposure changes)
//      and keep the best estimate seen, so no step can make a patch worse. On
//      the finest searched level (half resolution at most), patches that are
//      still poorly matched get a wide "rescue" search: thin, fast objects
//      vanish from coarse levels, so nothing coarser can hand their motion down.
//   2. densify: each pixel takes the average of the flows of the patches that
//      cover it, weighted by how well each flow explains that very pixel
//      (inverse photometric error). Patches straddling a motion boundary thus
//      contribute their flow only to the side they match.
//
// Draft and Normal stop at quarter and half resolution and upsample to full
// resolution photometrically (each pixel weighs the four nearest coarse flows
// by how well they match it). Finally, full-resolution refinement passes let
// each pixel adopt a neighbour's vector (8 directions, a few pixels away) when
// it matches a 3×3 window better, which snaps motion boundaries to image edges.
// All passes are local, fixed-size gathers: they map one-to-one to GPU kernels.
//
// The GPU kernels (plugin/gpu/smear_kernels.inc) implement the same passes with
// the same arithmetic; tests/test_gpu_path.cpp runs them on the CPU and compares.
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

namespace smear {

// --- Images --------------------------------------------------------------------

struct Pixel {
    float r = 0, g = 0, b = 0, a = 0;  // premultiplied
};

struct Vec2 {
    float x = 0, y = 0;
};

// Internal four-channel cell (pyramid levels, motion and tile buffers).
struct Vec4 {
    float x = 0, y = 0, z = 0, w = 0;
};

// A width × height grid of cells, row-major, pixel centres at integer coordinates.
template <class T>
struct Grid {
    int width = 0, height = 0;
    std::vector<T> px;

    Grid() = default;
    Grid(int w, int h) : width(w), height(h), px(size_t(w) * size_t(h)) {}
    T& at(int x, int y) { return px[size_t(y) * size_t(width) + size_t(x)]; }
    const T& at(int x, int y) const { return px[size_t(y) * size_t(width) + size_t(x)]; }
};

// Float RGBA, premultiplied (0–1 for 8/16 bpc, may exceed 1 at 32 bpc).
using Image = Grid<Pixel>;

// Per-pixel displacement in pixels: pixel p of `from` is at p + flow(p) in `to`.
using FlowField = Grid<Vec2>;

// --- Settings ------------------------------------------------------------------

enum class Quality { Draft = 1, Normal = 2, Best = 3 };

struct FlowSettings {
    // Motion Sensitivity, 0–100: higher follows finer detail (smaller patches),
    // lower gives smoother, steadier vectors (larger patches).
    double sensitivity = 50;
    Quality quality = Quality::Normal;
};

// Everything the flow passes need, derived from the settings and the frame size.
struct FlowPlan {
    int coarsest = 0;      // pyramid level the search starts at (0 = full resolution)
    int finest = 0;        // last level searched; upsampled to full resolution after
    int patch = 8;         // patch side, pixels
    int stride = 4;        // distance between patch origins
    int iterations = 12;   // Gauss-Newton steps per patch
    int search_radius = 3; // integer search around the initial flow before the steps, pixels
    int rescue_radius = 48;     // finest level: wide search (full-resolution pixels) for patches still poorly matched…
    float rescue_ratio = 0.5f;  // …i.e. with a residual above this fraction of their variance…
    float rescue_min_var = 1e-4f;  // …and enough texture (variance per pixel) to be matched at all
    int refine_steps[3] = {4, 1, 0};  // full-resolution boundary refinement passes (0 = off); Best adds a third
    float photo_eps = 0.02f;    // densify / upsample: errors below this weigh the same
    float max_step = 4.0f;      // longest single Gauss-Newton step, pixels
    float min_det = 1e-9f;      // patches with less texture keep their initial flow
};

// Size of pyramid level `level` along one axis (each level halves, rounding up).
inline int level_size(int full, int level) {
    int s = full;
    for (int l = 0; l < level; ++l) s = (s + 1) / 2;
    return std::max(1, s);
}

inline FlowPlan plan_flow(const FlowSettings& fs, int width, int height) {
    FlowPlan p;
    const double s = std::clamp(fs.sensitivity / 100.0, 0.0, 1.0);
    p.patch = int(std::lround(12.0 - 6.0 * s));  // 12 px (steady) .. 6 px (detailed)
    int finest = 1;
    switch (fs.quality) {
        case Quality::Draft:
            finest = 2;
            p.iterations = 8;
            p.refine_steps[1] = 0;
            p.rescue_radius = 0;
            p.stride = std::max(2, p.patch / 2);
            break;
        case Quality::Normal:
            finest = 1;
            p.iterations = 12;
            p.stride = std::max(2, p.patch / 2);
            break;
        case Quality::Best:
            finest = 0;
            p.iterations = 16;
            p.refine_steps[2] = 1;
            p.stride = std::max(2, p.patch / 3);
            break;
    }
    // Coarsest level: the smallest one still at least two patches across, so
    // motions of up to ~patch · 2^coarsest pixels are found.
    const int min_side = std::min(width, height);
    int coarsest = 0;
    while ((min_side >> (coarsest + 1)) >= 2 * p.patch) ++coarsest;
    p.coarsest = coarsest;
    p.finest = std::min(finest, coarsest);
    return p;
}

// Patch grid of one level.
struct PatchGrid {
    int patch = 0, stride = 0, nx = 0, ny = 0;
};

inline PatchGrid patch_grid(const FlowPlan& p, int w, int h) {
    PatchGrid g;
    g.patch = std::max(1, std::min(p.patch, std::min(w, h)));
    g.stride = std::max(1, p.stride);
    g.nx = (w - g.patch + g.stride - 1) / g.stride + 1;
    g.ny = (h - g.patch + g.stride - 1) / g.stride + 1;
    return g;
}

// --- Parallel loops --------------------------------------------------------------

// Shared across the process: After Effects renders several frames at once, so
// each loop takes its share of the cores.
inline std::atomic<int>& active_passes() {
    static std::atomic<int> n{0};
    return n;
}

template <class Fn>
void parallel_for(int count, Fn&& fn) {
    struct Guard {
        int concurrent = ++active_passes();
        ~Guard() { --active_passes(); }
    } guard;
    const int cores = std::max(1, int(std::thread::hardware_concurrency()));
    const int share = std::max(1, cores / std::max(1, guard.concurrent));
    const int workers = std::max(1, std::min(count / 4, share));
    if (workers <= 1) {
        for (int i = 0; i < count; ++i) fn(i);
        return;
    }
    std::atomic<int> next{0};
    auto run = [&] {
        for (int i; (i = next.fetch_add(1)) < count;) fn(i);
    };
    std::vector<std::thread> pool;
    pool.reserve(size_t(workers - 1));
    for (int t = 1; t < workers; ++t) pool.emplace_back(run);
    run();
    for (auto& t : pool) t.join();
}

// --- Shared helpers (mirrored one-to-one in smear_kernels.inc) -----------------

inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
inline float clampf(float v, float lo, float hi) { return std::fmin(std::fmax(v, lo), hi); }
inline float smoothstepf(float a, float b, float x) {
    const float t = clampf((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// Brightness used for matching: luma of the premultiplied colour plus some
// alpha, so shapes on transparent layers are tracked too.
inline float gray(const Pixel& p) { return 0.75f * (0.299f * p.r + 0.587f * p.g + 0.114f * p.b) + 0.25f * p.a; }

inline const Pixel& fetch_clamped(const Image& img, int x, int y) {
    return img.at(clampi(x, 0, img.width - 1), clampi(y, 0, img.height - 1));
}

// Bilinear weights and clamped corners for a position on a w × h grid.
struct Bilinear {
    int x0, y0, x1, y1;
    float fx, fy;
};
inline Bilinear bilinear(float x, float y, int w, int h) {
    x = clampf(x, -1.0f, float(w));  // also tames NaN-free runaway values
    y = clampf(y, -1.0f, float(h));
    const float fx0 = std::floor(x), fy0 = std::floor(y);
    Bilinear b;
    b.fx = x - fx0;
    b.fy = y - fy0;
    const int ix = int(fx0), iy = int(fy0);
    b.x0 = clampi(ix, 0, w - 1);
    b.x1 = clampi(ix + 1, 0, w - 1);
    b.y0 = clampi(iy, 0, h - 1);
    b.y1 = clampi(iy + 1, 0, h - 1);
    return b;
}

inline Vec2 sample_flow(const FlowField& f, float x, float y) {
    const Bilinear b = bilinear(x, y, f.width, f.height);
    const Vec2 p00 = f.at(b.x0, b.y0), p10 = f.at(b.x1, b.y0), p01 = f.at(b.x0, b.y1), p11 = f.at(b.x1, b.y1);
    const float tx = p00.x + (p10.x - p00.x) * b.fx, bx = p01.x + (p11.x - p01.x) * b.fx;
    const float ty = p00.y + (p10.y - p00.y) * b.fx, by = p01.y + (p11.y - p01.y) * b.fx;
    return {tx + (bx - tx) * b.fy, ty + (by - ty) * b.fy};
}

inline Pixel sample_image(const Image& img, float x, float y) {
    const Bilinear b = bilinear(x, y, img.width, img.height);
    const Pixel &p00 = img.at(b.x0, b.y0), &p10 = img.at(b.x1, b.y0);
    const Pixel &p01 = img.at(b.x0, b.y1), &p11 = img.at(b.x1, b.y1);
    auto mix = [&](float a, float c, float d, float e) {
        const float top = a + (c - a) * b.fx, bottom = d + (e - d) * b.fx;
        return top + (bottom - top) * b.fy;
    };
    return {mix(p00.r, p10.r, p01.r, p11.r), mix(p00.g, p10.g, p01.g, p11.g), mix(p00.b, p10.b, p01.b, p11.b),
            mix(p00.a, p10.a, p01.a, p11.a)};
}

// --- Flow passes ---------------------------------------------------------------

// A pyramid level: x = brightness of `from`, y = brightness of `to`,
// z, w = gradient of `from` (central differences).
using Level = Grid<Vec4>;

// Level 0 cell (x, y) from the two images.
inline Vec4 gray_cell(const Image& from, const Image& to, int x, int y) {
    const float l = gray(fetch_clamped(from, x - 1, y)), r = gray(fetch_clamped(from, x + 1, y));
    const float u = gray(fetch_clamped(from, x, y - 1)), d = gray(fetch_clamped(from, x, y + 1));
    return {gray(fetch_clamped(from, x, y)), gray(fetch_clamped(to, x, y)), 0.5f * (r - l), 0.5f * (d - u)};
}

// [1 3 3 1] / 8 tent of the finer level around coarse cell (x, y); x/y channels.
inline Vec2 tent(const Level& src, int x, int y) {
    const float wt[4] = {0.125f, 0.375f, 0.375f, 0.125f};
    float a = 0, b = 0;
    for (int j = 0; j < 4; ++j) {
        const int sy = clampi(2 * y - 1 + j, 0, src.height - 1);
        for (int i = 0; i < 4; ++i) {
            const int sx = clampi(2 * x - 1 + i, 0, src.width - 1);
            const float k = wt[i] * wt[j];
            const Vec4& s = src.at(sx, sy);
            a += k * s.x;
            b += k * s.y;
        }
    }
    return {a, b};
}

inline Vec4 downsample_cell(const Level& src, int w, int h, int x, int y) {
    const Vec2 c = tent(src, x, y);
    const float l = tent(src, clampi(x - 1, 0, w - 1), y).x, r = tent(src, clampi(x + 1, 0, w - 1), y).x;
    const float u = tent(src, x, clampi(y - 1, 0, h - 1)).x, d = tent(src, x, clampi(y + 1, 0, h - 1)).x;
    return {c.x, c.y, 0.5f * (r - l), 0.5f * (d - u)};
}

// Brightness of `to` (the y channel) at a fractional position, clamped to the edges.
inline float sample_to(const Level& L, float x, float y) {
    const Bilinear b = bilinear(x, y, L.width, L.height);
    const float p00 = L.at(b.x0, b.y0).y, p10 = L.at(b.x1, b.y0).y;
    const float p01 = L.at(b.x0, b.y1).y, p11 = L.at(b.x1, b.y1).y;
    const float top = p00 + (p10 - p00) * b.fx, bottom = p01 + (p11 - p01) * b.fx;
    return top + (bottom - top) * b.fy;
}

inline int patch_origin(int i, int stride, int patch, int size) { return std::min(i * stride, size - patch); }

// Neighbour directions for the candidate flows.
constexpr int kOffX[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
constexpr int kOffY[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

// Sum of squared mean-normalised differences between the patch at (ox, oy) and
// `to` displaced by u. `mt`, `var_t`: mean and summed squared deviation of the template.
inline float patch_cost(const Level& L, int ox, int oy, int ps, float mt, float var_t, Vec2 u) {
    float sw = 0, sww = 0, swt = 0;
    for (int ky = 0; ky < ps; ++ky)
        for (int kx = 0; kx < ps; ++kx) {
            const float wv = sample_to(L, float(ox + kx) + u.x, float(oy + ky) + u.y);
            sw += wv;
            sww += wv * wv;
            swt += wv * (L.at(ox + kx, oy + ky).x - mt);
        }
    return (sww - sw * sw / float(ps * ps)) - 2.0f * swt + var_t;
}

// Integer displacements within ±r of `around` (rounded): keeps the best one in
// best/best_e (sum of squared mean-normalised differences).
inline void integer_search(const Level& L, int ox, int oy, int ps, float mt, float var_t, Vec2 around, int r,
                           Vec2& best, float& best_e) {
    const float inv_n = 1.0f / float(ps * ps);
    const int rx = int(std::floor(around.x + 0.5f)), ry = int(std::floor(around.y + 0.5f));
    for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx) {
            float sw = 0, sww = 0, swt = 0;
            for (int ky = 0; ky < ps; ++ky) {
                const int sy = clampi(oy + ky + ry + dy, 0, L.height - 1);
                for (int kx = 0; kx < ps; ++kx) {
                    const int sx = clampi(ox + kx + rx + dx, 0, L.width - 1);
                    const float wv = L.at(sx, sy).y;
                    sw += wv;
                    sww += wv * wv;
                    swt += wv * (L.at(ox + kx, oy + ky).x - mt);
                }
            }
            const float e = (sww - sw * sw * inv_n) - 2.0f * swt + var_t;
            if (e < best_e) {
                best_e = e;
                best = {float(rx + dx), float(ry + dy)};
            }
        }
}

// Flow of patch (i, j). `coarse`: the dense flow of the next coarser level, or null.
inline Vec2 patch_search_one(const Level& L, const FlowField* coarse, const FlowPlan& plan, const PatchGrid& g,
                             int radius, int rescue, int i, int j) {
    const int ps = g.patch, n = ps * ps;
    const int ox = patch_origin(i, g.stride, ps, L.width), oy = patch_origin(j, g.stride, ps, L.height);
    const float cx = float(ox) + float(ps - 1) * 0.5f, cy = float(oy) + float(ps - 1) * 0.5f;
    Vec2 u;
    if (coarse) {
        const Vec2 c = sample_flow(*coarse, cx * 0.5f - 0.25f, cy * 0.5f - 0.25f);
        u = {2.0f * c.x, 2.0f * c.y};
    }
    // Template statistics: means, then the Hessian of the mean-normalised patch.
    float st = 0, sgx = 0, sgy = 0;
    for (int ky = 0; ky < ps; ++ky)
        for (int kx = 0; kx < ps; ++kx) {
            const Vec4& t = L.at(ox + kx, oy + ky);
            st += t.x;
            sgx += t.z;
            sgy += t.w;
        }
    const float inv_n = 1.0f / float(n);
    const float mt = st * inv_n, mgx = sgx * inv_n, mgy = sgy * inv_n;
    float hxx = 0, hxy = 0, hyy = 0, bx0 = 0, by0 = 0, var_t = 0;
    for (int ky = 0; ky < ps; ++ky)
        for (int kx = 0; kx < ps; ++kx) {
            const Vec4& t = L.at(ox + kx, oy + ky);
            const float gx = t.z - mgx, gy = t.w - mgy, d = t.x - mt;
            hxx += gx * gx;
            hxy += gx * gy;
            hyy += gy * gy;
            bx0 += gx * d;
            by0 += gy * d;
            var_t += d * d;
        }
    const float det = hxx * hyy - hxy * hxy;
    if (!(det > plan.min_det * float(n) * float(n))) return u;  // no texture: keep the initial flow
    const float inv_det = 1.0f / det;

    // Candidates: the coarse flow at the patch centre and one patch away in
    // each direction; near motion boundaries, the side the patch mostly
    // belongs to wins.
    if (coarse) {
        float ce = patch_cost(L, ox, oy, ps, mt, var_t, u);
        for (int k = 0; k < 8; ++k) {
            const float px = cx + float(kOffX[k] * ps), py = cy + float(kOffY[k] * ps);
            const Vec2 c = sample_flow(*coarse, px * 0.5f - 0.25f, py * 0.5f - 0.25f);
            const Vec2 v = {2.0f * c.x, 2.0f * c.y};
            const float e = patch_cost(L, ox, oy, ps, mt, var_t, v);
            if (e < ce) {
                ce = e;
                u = v;
            }
        }
    }

    // Integer search around the initial flow: widens the capture range of each
    // level from the patch's own texture scale to ±radius pixels, so motions
    // too fast for the coarse levels are still found.
    Vec2 best = u;
    float best_e = 3.0e38f;
    if (radius > 0) {
        integer_search(L, ox, oy, ps, mt, var_t, u, radius, best, best_e);
        u = best;
    }
    // Rescue: a patch still poorly matched gets a much wider search.
    if (rescue > 0 && best_e > plan.rescue_ratio * var_t && var_t > plan.rescue_min_var * float(n)) {
        integer_search(L, ox, oy, ps, mt, var_t, u, rescue, best, best_e);
        u = best;
    }
    for (int it = 0; it <= plan.iterations; ++it) {
        float sw = 0, sww = 0, swt = 0, bx = 0, by = 0;
        for (int ky = 0; ky < ps; ++ky)
            for (int kx = 0; kx < ps; ++kx) {
                const Vec4& t = L.at(ox + kx, oy + ky);
                const float wv = sample_to(L, float(ox + kx) + u.x, float(oy + ky) + u.y);
                sw += wv;
                sww += wv * wv;
                swt += wv * (t.x - mt);
                bx += (t.z - mgx) * wv;
                by += (t.w - mgy) * wv;
            }
        // Sum of squared mean-normalised differences at u.
        const float e = (sww - sw * sw * inv_n) - 2.0f * swt + var_t;
        if (e < best_e) {
            best_e = e;
            best = u;
        }
        if (it == plan.iterations) break;
        bx -= bx0;
        by -= by0;
        float dx = (hyy * bx - hxy * by) * inv_det, dy = (hxx * by - hxy * bx) * inv_det;
        const float len = std::sqrt(dx * dx + dy * dy);
        if (len > plan.max_step) {
            dx *= plan.max_step / len;
            dy *= plan.max_step / len;
        }
        u.x -= dx;
        u.y -= dy;
    }
    return best;
}

// Integer search radius at pyramid level `level`: wide on the two coarsest
// levels (where it is cheap and nothing coarser has found the motion yet).
inline int search_radius(const FlowPlan& p, int level) {
    return level >= p.coarsest - 1 ? std::max(p.search_radius, p.patch) : p.search_radius;
}

// Rescue search radius at `level`, in level pixels: only on the finest searched
// level, or half resolution for Best (0 = none). Thin, fast objects vanish from the coarse levels, so
// nothing coarser can hand their motion down.
inline int rescue_radius(const FlowPlan& p, int level) {
    const int at = std::max(p.finest, 1);  // never at full resolution: too costly for what it adds
    return (p.rescue_radius > 0 && level == at && level < p.coarsest - 1)
               ? std::max(p.search_radius + 1, p.rescue_radius >> level)
               : 0;
}

inline FlowField patch_search(const Level& L, const FlowField* coarse, const FlowPlan& plan, int level) {
    const PatchGrid g = patch_grid(plan, L.width, L.height);
    const int radius = search_radius(plan, level), rescue = rescue_radius(plan, level);
    FlowField out(g.nx, g.ny);
    parallel_for(g.ny, [&](int j) {
        for (int i = 0; i < g.nx; ++i) out.at(i, j) = patch_search_one(L, coarse, plan, g, radius, rescue, i, j);
    });
    return out;
}

// Weight of flow `u` at level pixel (x, y): inverse photometric error.
inline float photo_weight(const Level& L, int x, int y, Vec2 u, float eps) {
    const float e = std::fabs(sample_to(L, float(x) + u.x, float(y) + u.y) - L.at(x, y).x);
    return 1.0f / std::fmax(e, eps);
}

inline Vec2 densify_one(const Level& L, const FlowField& patches, const FlowPlan& plan, const PatchGrid& g, int x,
                        int y) {
    const int ps = g.patch, st = g.stride;
    const int i0 = std::max(0, (x - ps + 1 + st - 1) / st), j0 = std::max(0, (y - ps + 1 + st - 1) / st);
    float sx = 0, sy = 0, sw = 0;
    for (int j = j0; j < g.ny; ++j) {
        const int oy = patch_origin(j, st, ps, L.height);
        if (oy > y) break;
        if (y >= oy + ps) continue;
        for (int i = i0; i < g.nx; ++i) {
            const int ox = patch_origin(i, st, ps, L.width);
            if (ox > x) break;
            if (x >= ox + ps) continue;
            const Vec2 u = patches.at(i, j);
            const float w = photo_weight(L, x, y, u, plan.photo_eps);
            sx += w * u.x;
            sy += w * u.y;
            sw += w;
        }
    }
    if (!(sw > 0.0f)) return {};
    return {sx / sw, sy / sw};
}

inline FlowField densify(const Level& L, const FlowField& patches, const FlowPlan& plan) {
    const PatchGrid g = patch_grid(plan, L.width, L.height);
    FlowField out(L.width, L.height);
    parallel_for(L.height, [&](int y) {
        for (int x = 0; x < L.width; ++x) out.at(x, y) = densify_one(L, patches, plan, g, x, y);
    });
    return out;
}

// Full-resolution flow at (x, y) from the flow of a level `scale` times smaller:
// the four nearest coarse vectors, weighted by bilinear position and by how
// well each one matches this pixel.
inline Vec2 upsample_one(const Level& L0, const FlowField& coarse, int scale, float eps, int x, int y) {
    const float inv = 1.0f / float(scale);
    const float qx = (float(x) + 0.5f) * inv - 0.5f, qy = (float(y) + 0.5f) * inv - 0.5f;
    const Bilinear b = bilinear(qx, qy, coarse.width, coarse.height);
    const int cxs[2] = {b.x0, b.x1}, cys[2] = {b.y0, b.y1};
    const float wxs[2] = {1.0f - b.fx, b.fx}, wys[2] = {1.0f - b.fy, b.fy};
    float sx = 0, sy = 0, sw = 0;
    for (int j = 0; j < 2; ++j)
        for (int i = 0; i < 2; ++i) {
            const Vec2 c = coarse.at(cxs[i], cys[j]);
            const Vec2 u = {c.x * float(scale), c.y * float(scale)};
            const float w = (wxs[i] * wys[j] + 1e-3f) * photo_weight(L0, x, y, u, eps);
            sx += w * u.x;
            sy += w * u.y;
            sw += w;
        }
    return {sx / sw, sy / sw};
}

// Boundary refinement at full resolution: each pixel keeps its flow or takes
// that of a neighbour `step` pixels away (8 directions), whichever best matches
// a 3×3 window around it. Motion boundaries snap to image edges.
inline float window_error(const Level& L0, int x, int y, Vec2 u) {
    float e = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const int px = clampi(x + dx, 0, L0.width - 1), py = clampi(y + dy, 0, L0.height - 1);
            e += std::fabs(sample_to(L0, float(px) + u.x, float(py) + u.y) - L0.at(px, py).x);
        }
    return e;
}

inline Vec2 refine_one(const Level& L0, const FlowField& flow, int step, int x, int y) {
    Vec2 best = flow.at(x, y);
    float best_e = window_error(L0, x, y, best);
    for (int k = 0; k < 8; ++k) {
        const int px = clampi(x + kOffX[k] * step, 0, flow.width - 1), py = clampi(y + kOffY[k] * step, 0, flow.height - 1);
        const Vec2 u = flow.at(px, py);
        if (std::fabs(u.x - best.x) + std::fabs(u.y - best.y) < 0.05f) continue;  // same vector: nothing to gain
        const float e = window_error(L0, x, y, u);
        if (e < best_e) {
            best_e = e;
            best = u;
        }
    }
    return best;
}

// Neighbour distances of the refinement passes, coarse to fine (0 = skipped).
struct RefineSteps {
    int step[3];
    const int* begin() const { return step; }
    const int* end() const { return step + 3; }
};
inline RefineSteps refine_steps(const FlowPlan& p) { return {{p.refine_steps[0], p.refine_steps[1], p.refine_steps[2]}}; }

// Dense flow from `from` to `to` (same size). `out` is resized to that size.
inline void compute_flow(const Image& from, const Image& to, FlowField& out, const FlowSettings& settings) {
    const int w = from.width, h = from.height;
    out = FlowField(w, h);
    if (w <= 0 || h <= 0 || to.width != w || to.height != h) return;
    const FlowPlan plan = plan_flow(settings, w, h);

    std::vector<Level> pyr(size_t(plan.coarsest) + 1);
    pyr[0] = Level(w, h);
    parallel_for(h, [&](int y) {
        for (int x = 0; x < w; ++x) pyr[0].at(x, y) = gray_cell(from, to, x, y);
    });
    for (int l = 1; l <= plan.coarsest; ++l) {
        const int lw = level_size(w, l), lh = level_size(h, l);
        Level& dst = pyr[size_t(l)];
        dst = Level(lw, lh);
        const Level& src = pyr[size_t(l) - 1];
        parallel_for(lh, [&](int y) {
            for (int x = 0; x < lw; ++x) dst.at(x, y) = downsample_cell(src, lw, lh, x, y);
        });
    }

    FlowField flow;
    for (int l = plan.coarsest; l >= plan.finest; --l) {
        const Level& L = pyr[size_t(l)];
        const FlowField patches = patch_search(L, l == plan.coarsest ? nullptr : &flow, plan, l);
        flow = densify(L, patches, plan);
    }
    if (plan.finest > 0) {
        const int scale = 1 << plan.finest;
        FlowField up(w, h);
        parallel_for(h, [&](int y) {
            for (int x = 0; x < w; ++x) up.at(x, y) = upsample_one(pyr[0], flow, scale, plan.photo_eps, x, y);
        });
        flow = std::move(up);
    }
    for (int step : refine_steps(plan)) {
        if (step <= 0) continue;
        FlowField next(w, h);
        parallel_for(h, [&](int y) {
            for (int x = 0; x < w; ++x) next.at(x, y) = refine_one(pyr[0], flow, step, x, y);
        });
        flow = std::move(next);
    }
    out = std::move(flow);
}

}  // namespace smear
