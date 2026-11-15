// crosstalk_core.h — the effect's controls and how they become a CrosstalkParams block.
//
// Independent of After Effects. The plugin reads the controls into Params,
// make_params() adds the frame (time, size, resolution, bit depth) and the
// renderers (crosstalk_cpu.h, plugin/gpu) run crosstalk_kernels.inc.
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

#include "crosstalk_types.h"

namespace crosstalk {

enum class Method { Shift = 1, Linear = 2, Noisy = 3, Sawtooth = 4, Wave = 5, WaveMotion = 6 };

// What shows where a displaced line leaves the layer.
enum class Edges { Transparent = 1, Mirror = 2, Tile = 3, Black = 4 };

struct Rgb {
    double r = 1, g = 1, b = 1;  // 0–1
};

// The settings of one field (the even or the odd band of rows).
struct Field {
    double size = 2;          // rows per band, full resolution
    double process = 68;      // %: 0 = original, 100 = tinted
    double luminance = -99;   // %, added brightness
    bool tint = true;
    Rgb tint_color{248 / 255.0, 1.0, 248 / 255.0};
    double noise = 3;         // %
    bool chroma = false;
    int decay = 1;            // echo taps (1 = off)
    bool smear = true;
    Method method = Method::Wave;
    double hadj = 0;          // Horiz. Adj.
    int comb = 0;             // Comb Filt. Level (column mask)
};

// Parameter values as shown in the Effect Controls panel. Defaults match the panel.
struct Params {
    double blend = 0;  // Blend With Original, %: cross-fade back to the source
    int seed = 0;
    Edges edges = Edges::Tile;
    double influence = 1000;
    Field even;
    Field odd = [] {
        Field f;
        f.process = 5;
        f.luminance = 4;
        f.tint_color = {248 / 255.0, 248 / 255.0, 248 / 255.0};
        f.noise = 6;
        f.decay = 2;
        f.comb = 1;
        return f;
    }();
};

// Where and when a frame is rendered.
struct Frame {
    long long time = 0;  // layer time, time_scale units
    long long time_scale = 30;
    int full_w = 0, full_h = 0;      // layer size, full resolution
    int canvas_w = 0, canvas_h = 0;  // the whole layer at the current resolution
    double ds_x = 1, ds_y = 1;       // current resolution / full resolution
    int bits = 8;
    int in_w = 0, in_h = 0, in_ox = 0, in_oy = 0;      // input buffer and its corner on the canvas
    int out_w = 0, out_h = 0, out_ox = 0, out_oy = 0;  // output buffer and its corner on the canvas
};

inline uint32_t mix32(uint32_t v) {
    v ^= v >> 16;
    v *= 0x7feb352du;
    v ^= v >> 15;
    v *= 0x846ca68bu;
    v ^= v >> 16;
    return v;
}

inline CrosstalkParams make_params(const Params& c, const Frame& f) {
    CrosstalkParams p{};
    p.canvas_w = std::max(1, f.canvas_w);
    p.canvas_h = std::max(1, f.canvas_h);
    p.ds_x = float(std::clamp(f.ds_x, 1e-3, 1.0));
    p.ds_y = float(std::clamp(f.ds_y, 1e-3, 1.0));
    p.inv_ds_x = 1.0f / p.ds_x;
    p.inv_ds_y = 1.0f / p.ds_y;
    p.full_w = f.full_w > 0 ? f.full_w : std::max(1, int(std::lround(p.canvas_w / double(p.ds_x))));
    p.full_h = f.full_h > 0 ? f.full_h : std::max(1, int(std::lround(p.canvas_h / double(p.ds_y))));
    p.in_w = f.in_w;
    p.in_h = f.in_h;
    p.in_pitch = f.in_w;
    p.in_ox = f.in_ox;
    p.in_oy = f.in_oy;
    p.out_w = f.out_w;
    p.out_h = f.out_h;
    p.out_pitch = f.out_w;
    p.out_ox = f.out_ox;
    p.out_oy = f.out_oy;

    // The noise changes every frame (and with the seed).
    const uint32_t seed = mix32(uint32_t(c.seed) ^ 0x63a1b5c7u);
    p.key = mix32(seed ^ mix32(uint32_t(f.time) ^ mix32(uint32_t(f.time >> 32) + 0x2545f491u) ^
                               mix32(uint32_t(f.time_scale) * 0x9e3779b9u)));
    p.edges = std::clamp(int(c.edges), 1, 4);
    p.blend = float(std::clamp(c.blend / 100.0, 0.0, 1.0));
    p.color_max = f.bits >= 32 ? 1e30f : 1.0f;
    p.influence = float(c.influence);

    const Field* fields[2] = {&c.even, &c.odd};
    for (int i = 0; i < 2; ++i) {
        const Field& d = *fields[i];
        p.size[i] = std::clamp(int(std::lround(d.size)), 1, 4000);
        p.process[i] = float(std::clamp(d.process / 100.0, 0.0, 1.0));
        p.lum[i] = float(d.luminance / 100.0);
        p.tint[i] = d.tint ? 1 : 0;
        p.tint_r[i] = float(d.tint_color.r);
        p.tint_g[i] = float(d.tint_color.g);
        p.tint_b[i] = float(d.tint_color.b);
        p.noise[i] = float(std::max(0.0, d.noise) / 100.0);
        p.chroma[i] = d.chroma ? 1 : 0;
        p.decay[i] = std::clamp(d.decay, 1, 32);
        p.smear[i] = d.smear ? 1 : 0;
        p.method[i] = std::clamp(int(d.method), 1, 6);
        p.hadj[i] = float(d.hadj);
        p.comb[i] = std::clamp(d.comb, 0, 255);
    }
    return p;
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

}  // namespace crosstalk
