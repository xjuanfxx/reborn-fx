// test_util.h — tiny test harness and synthetic two-layer scenes with known motion.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>

#include "smear_render.h"

static int failures = 0;

#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                 \
        }                                                               \
    } while (0)

#define CHECK_LT(a, b)                                                                                  \
    do {                                                                                                \
        const double va = double(a), vb = double(b);                                                    \
        if (!(va < vb)) {                                                                               \
            std::printf("FAIL %s:%d: %s < %s (%g vs %g)\n", __FILE__, __LINE__, #a, #b, va, vb);         \
            ++failures;                                                                                 \
        }                                                                                               \
    } while (0)

namespace test {

inline float lattice(int x, int y, uint32_t seed) {
    uint32_t h = uint32_t(x) * 0x8da6b343u ^ uint32_t(y) * 0xd8163841u ^ seed * 0xcb1ab31fu;
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    return float(h & 0xffffu) / 65535.0f;
}

// Smooth value noise, features ~`scale` pixels.
inline float noise(float x, float y, float scale, uint32_t seed) {
    x /= scale;
    y /= scale;
    const float fx0 = std::floor(x), fy0 = std::floor(y);
    const int ix = int(fx0), iy = int(fy0);
    float fx = x - fx0, fy = y - fy0;
    fx = fx * fx * (3 - 2 * fx);
    fy = fy * fy * (3 - 2 * fy);
    const float a = lattice(ix, iy, seed), b = lattice(ix + 1, iy, seed);
    const float c = lattice(ix, iy + 1, seed), d = lattice(ix + 1, iy + 1, seed);
    const float top = a + (b - a) * fx, bottom = c + (d - c) * fx;
    return top + (bottom - top) * fy;
}

inline smear::Pixel texture(float x, float y, uint32_t seed) {
    auto ch = [&](uint32_t s) {
        return 0.6f * noise(x, y, 9.0f, seed + s) + 0.4f * noise(x, y, 4.0f, seed + s + 50);
    };
    return {ch(0), ch(1), ch(2), 1.0f};
}

// Background translating by (bvx, bvy) px/frame; optional square of half-size
// `half` centred at (cx, cy) at frame 0, translating by (fvx, fvy).
struct Scene {
    int w = 96, h = 72;
    float bvx = 0, bvy = 0;
    bool square = false;
    float cx = 48, cy = 36, half = 14, fvx = 0, fvy = 0;

    bool in_square(float x, float y, float t) const {
        return square && std::fabs(x - (cx + fvx * t)) <= half && std::fabs(y - (cy + fvy * t)) <= half;
    }
    smear::Pixel colour(float x, float y, float t) const {
        if (in_square(x, y, t)) return texture(x - fvx * t + 300.0f, y - fvy * t, 7);
        return texture(x - bvx * t, y - bvy * t, 3);
    }
    // 2×2 supersampled frame at time t.
    smear::Image frame(float t) const {
        smear::Image img(w, h);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                smear::Pixel s{};
                for (int j = 0; j < 2; ++j)
                    for (int i = 0; i < 2; ++i) {
                        const smear::Pixel p = colour(float(x) + (float(i) - 0.5f) * 0.5f, float(y) + (float(j) - 0.5f) * 0.5f, t);
                        s.r += p.r * 0.25f;
                        s.g += p.g * 0.25f;
                        s.b += p.b * 0.25f;
                        s.a += p.a * 0.25f;
                    }
                img.at(x, y) = s;
            }
        return img;
    }
    // True flow from frame 0 to frame t.
    smear::FlowField flow(float t) const {
        smear::FlowField f(w, h);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                f.at(x, y) = in_square(float(x), float(y), 0) ? smear::Vec2{fvx * t, fvy * t} : smear::Vec2{bvx * t, bvy * t};
        return f;
    }
};

inline double mean_abs_diff(const smear::Image& a, const smear::Image& b) {
    double s = 0;
    for (size_t i = 0; i < a.px.size(); ++i)
        s += std::fabs(a.px[i].r - b.px[i].r) + std::fabs(a.px[i].g - b.px[i].g) + std::fabs(a.px[i].b - b.px[i].b) +
             std::fabs(a.px[i].a - b.px[i].a);
    return s / (4.0 * double(a.px.size()));
}

inline double max_abs_diff(const smear::Image& a, const smear::Image& b) {
    double m = 0;
    for (size_t i = 0; i < a.px.size(); ++i) {
        m = std::fmax(m, std::fabs(a.px[i].r - b.px[i].r));
        m = std::fmax(m, std::fabs(a.px[i].g - b.px[i].g));
        m = std::fmax(m, std::fabs(a.px[i].b - b.px[i].b));
        m = std::fmax(m, std::fabs(a.px[i].a - b.px[i].a));
    }
    return m;
}

inline double mean_flow_diff(const smear::FlowField& a, const smear::FlowField& b) {
    double s = 0;
    for (size_t i = 0; i < a.px.size(); ++i) s += std::fabs(a.px[i].x - b.px[i].x) + std::fabs(a.px[i].y - b.px[i].y);
    return s / (2.0 * double(a.px.size()));
}

}  // namespace test
