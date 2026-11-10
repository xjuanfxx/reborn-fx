// tremor_render.h — CPU renderer.
//
// Works on float RGBA images (0–1 for 8/16 bpc) holding the whole layer. The
// GPU kernels (plugin/gpu/tremor_kernels.inc) implement the same three passes;
// tools/gpu_check compares both.
//
//   1. deinterlace: optional field filter on the source.
//   2. rotate:      rotates R, G and B by their own angle around the center.
//   3. finish:      zooms and shifts the rotated image, applies the color
//                   offsets, composites it over the source if asked and
//                   cross-fades back to the source (Blend).
//
// Pixels are premultiplied, as After Effects provides them.
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <thread>
#include <vector>

#include "tremor_core.h"

namespace tremor {

struct Pixel {
    float r = 0, g = 0, b = 0, a = 0;
};

struct Image {
    int width = 0, height = 0;
    std::vector<Pixel> px;

    Image() = default;
    Image(int w, int h) : width(w), height(h), px(size_t(w) * size_t(h)) {}
    Pixel& at(int x, int y) { return px[size_t(y) * size_t(width) + size_t(x)]; }
    const Pixel& at(int x, int y) const { return px[size_t(y) * size_t(width) + size_t(x)]; }
};

// How the rendered frame is combined with the source.
struct Look {
    Deinterlace deinterlace = Deinterlace::None;
    Edges edges = Edges::Transparent;
    bool alpha_composite = false;  // the source shows through where the moved picture is transparent
    float adjust[3] = {0, 0, 0};   // added to R, G, B of the moved picture (0–1 units)
    float blend = 0.0f;            // Blend With Original / 100
    // 8 and 16 bpc clip at 1; 32 bpc keeps color values above 1.
    float color_max = 1.0f;
};

inline Look make_look(const Params& p, int bits_per_channel) {
    Look l;
    l.deinterlace = p.deinterlace;
    l.edges = p.edges;
    l.alpha_composite = p.alpha_composite;
    l.adjust[0] = float(p.r_adjust / 100.0);
    l.adjust[1] = float(p.g_adjust / 100.0);
    l.adjust[2] = float(p.b_adjust / 100.0);
    l.blend = float(std::clamp(p.blend / 100.0, 0.0, 1.0));
    l.color_max = bits_per_channel >= 32 ? std::numeric_limits<float>::infinity() : 1.0f;
    return l;
}

// Parallel loops shared across the process (After Effects renders several frames at once).
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
    const int workers = std::max(1, std::min(count / 16, share));
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

// Pixel (x, y), possibly outside the image, according to the edge mode.
inline Pixel fetch(const Image& img, int x, int y, Edges edges) {
    const int w = img.width, h = img.height;
    if (x < 0 || y < 0 || x >= w || y >= h) {
        switch (edges) {
            case Edges::Transparent: return {};
            case Edges::Black: return {0, 0, 0, 1};
            case Edges::Tile:
                x = ((x % w) + w) % w;
                y = ((y % h) + h) % h;
                break;
            case Edges::Mirror: {
                const int pw = 2 * w, ph = 2 * h;
                x = ((x % pw) + pw) % pw;
                y = ((y % ph) + ph) % ph;
                if (x >= w) x = pw - 1 - x;
                if (y >= h) y = ph - 1 - y;
                break;
            }
        }
    }
    return img.at(x, y);
}

// Bilinear sample; pixel centers sit at integer coordinates.
inline Pixel sample(const Image& img, double x, double y, Edges edges) {
    if (!(std::fabs(x) < 1e8 && std::fabs(y) < 1e8))  // collapsed zoom or NaN
        return edges == Edges::Black ? Pixel{0, 0, 0, 1} : Pixel{};
    const double fx0 = std::floor(x), fy0 = std::floor(y);
    const int x0 = int(fx0), y0 = int(fy0);
    const float fx = float(x - fx0), fy = float(y - fy0);
    const Pixel p00 = fetch(img, x0, y0, edges), p10 = fetch(img, x0 + 1, y0, edges);
    const Pixel p01 = fetch(img, x0, y0 + 1, edges), p11 = fetch(img, x0 + 1, y0 + 1, edges);
    auto mix = [&](float a, float b, float c, float d) {
        const float top = a + (b - a) * fx, bottom = c + (d - c) * fx;
        return top + (bottom - top) * fy;
    };
    return {mix(p00.r, p10.r, p01.r, p11.r), mix(p00.g, p10.g, p01.g, p11.g), mix(p00.b, p10.b, p01.b, p11.b),
            mix(p00.a, p10.a, p01.a, p11.a)};
}

// Field filters on the color channels, always reading the unfiltered rows.
inline void deinterlace(const Image& src, Image& dst, Deinterlace mode) {
    const int w = src.width, h = src.height;
    parallel_for(h, [&](int y) {
        int rebuild = 0;  // 1: average of the rows above and below, 2: [1 2 1] / 4
        if (y > 0 && y < h - 1) {
            if (mode == Deinterlace::LowerOnly && y % 2 == 0) rebuild = 1;
            if (mode == Deinterlace::UpperOnly && y % 2 == 1) rebuild = 1;
            if (mode == Deinterlace::Merge) rebuild = 2;
        }
        for (int x = 0; x < w; ++x) {
            Pixel m = src.at(x, y);
            if (rebuild) {
                const Pixel &a = src.at(x, y - 1), &b = src.at(x, y + 1);
                const float wa = rebuild == 1 ? 0.5f : 0.25f, wm = rebuild == 1 ? 0.0f : 0.5f;
                m.r = (a.r + b.r) * wa + m.r * wm;
                m.g = (a.g + b.g) * wa + m.g * wm;
                m.b = (a.b + b.b) * wa + m.b * wm;
            }
            dst.at(x, y) = m;
        }
    });
}

// cos/sin of the R, G and B angles.
struct Turns {
    double c[3], s[3];
};
inline Turns turns(const FrameState& st) {
    Turns t;
    for (int k = 0; k < 3; ++k) {
        const double a = (st.rotation_deg + k * st.separation_deg) * (kPi / 180.0);
        t.c[k] = std::cos(a);
        t.s[k] = std::sin(a);
    }
    return t;
}

inline void rotate(const Image& src, Image& dst, const FrameState& st, Edges edges) {
    const Turns t = turns(st);
    const double cx = st.center_x, cy = st.center_y;
    parallel_for(dst.height, [&](int y) {
        const double dy = y - cy;
        for (int x = 0; x < dst.width; ++x) {
            const double dx = x - cx;
            Pixel p[3];
            for (int k = 0; k < 3; ++k)
                p[k] = sample(src, cx + dx * t.c[k] - dy * t.s[k], cy + dx * t.s[k] + dy * t.c[k], edges);
            dst.at(x, y) = {p[0].r, p[1].g, p[2].b, (p[0].a + p[1].a + p[2].a) / 3.0f};
        }
    });
}

// Output pixel (x, y) sits at (x + ox, y + oy) on the source/rotated images.
inline void finish(const Image& rotated, const Image& source, Image& out, int ox, int oy, const FrameState& st,
                   const Look& look) {
    const double inv = 1.0 / (st.zoom + 1.0);
    const float hi = look.color_max, keep = 1.0f - look.blend;
    parallel_for(out.height, [&](int y) {
        const int sy = y + oy;
        for (int x = 0; x < out.width; ++x) {
            const int sx = x + ox;
            const bool inside = sx >= 0 && sy >= 0 && sx < source.width && sy < source.height;
            const Pixel src = inside ? source.at(sx, sy) : Pixel{};
            if (keep <= 0.0f) {  // Blend 100%: the source as is
                out.at(x, y) = src;
                continue;
            }
            Pixel o = sample(rotated, st.center_x + (sx - st.center_x) * inv - st.shift_x,
                             st.center_y + (sy - st.center_y) * inv - st.shift_y, look.edges);
            // Color offsets scale with coverage, so transparent areas stay clear.
            o.r = std::clamp(o.r + look.adjust[0] * o.a, 0.0f, hi);
            o.g = std::clamp(o.g + look.adjust[1] * o.a, 0.0f, hi);
            o.b = std::clamp(o.b + look.adjust[2] * o.a, 0.0f, hi);
            if (look.alpha_composite) {  // moved picture over the source
                const float t = 1.0f - o.a;
                o = {o.r + src.r * t, o.g + src.g * t, o.b + src.b * t, o.a + src.a * t};
            }
            out.at(x, y) = {o.r * keep + src.r * look.blend, o.g * keep + src.g * look.blend,
                            o.b * keep + src.b * look.blend, o.a * keep + src.a * look.blend};
        }
    });
}

// `canvas`: the whole layer. `out`: the requested area, at (ox, oy) on the canvas.
inline void render_frame(const Image& canvas, Image& out, int ox, int oy, const FrameState& st, const Look& look) {
    Image source(canvas.width, canvas.height);
    deinterlace(canvas, source, look.deinterlace);
    Image rotated(source.width, source.height);
    if (look.blend < 1.0f) rotate(source, rotated, st, look.edges);
    finish(rotated, source, out, ox, oy, st, look);
}

}  // namespace tremor
