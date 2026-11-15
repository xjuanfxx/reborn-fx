// reception_core.h — the effect's controls and how they become a ReceptionParams block.
//
// Independent of After Effects. The plugin reads the controls into Params,
// make_params() adds the frame (time, size, resolution, bit depth) and the
// values that evolve from frame to frame (distortion strength, ghost offsets,
// bar position, vertical roll); the renderers (reception_cpu.h, plugin/gpu)
// run reception_kernels.inc.
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

#include "reception_types.h"

namespace reception {

// What shows where the distortion, the ghosts or the roll reach past the layer.
enum class Edges { Transparent = 1, Mirror = 2, Tile = 3, Black = 4 };

struct Rgb {
    double r = 1, g = 1, b = 1;  // 0–1
};

// Parameter values as shown in the Effect Controls panel. Defaults match the panel.
struct Params {
    double blend = 0;  // Blend With Original, %: cross-fade back to the source
    int seed = 0;
    Edges edges = Edges::Tile;

    // Color
    Rgb tint{233 / 255.0, 233 / 255.0, 233 / 255.0};
    double lift = 0;           // Brightness Lift, 8-bit levels
    double bright_var = 10;    // Bright Variance
    double contrast = 0;       // -50..50
    double contrast_var = 0;   // Contrast Variance

    // Noise
    double noise_max = 30;     // %
    double noise_var = 5;
    double micro = 30;         // Micro Noise, %
    double noise_width = 15;   // full-resolution pixels
    double noise_tempo = 20;   // full-resolution rows

    // Distortion
    double dist_min = 50, dist_max = 100;  // % of the amplitudes
    double tempo_min = 40, tempo_max = 60; // frames between strength changes
    double amp1 = 40, amp2 = 15;           // full-resolution pixels
    double freq1 = 2.5, freq2 = 7;
    double speed1 = 10, speed2 = 20;
    double offset_var = 2;

    // Glow
    double glow_size = 1;
    double glow_intensity = 10;
    double glow_threshold = 84;  // %

    // Ghosting
    double ghost_off1 = 10, ghost_off2 = 10, ghost_off3 = 10;  // full-resolution pixels
    double w_primary = 40, w_ghost1 = 10, w_ghost2 = 10, w_ghost3 = 10;  // %

    // Bars
    double bar_height = 40;
    double bar_ripple = 3;
    double bar_dark = 70;

    // Vertical Hold
    double roll_prob = 10;    // % chance that a hold period rolls
    double roll_length = 80;  // frames
    double roll_speed = 20;   // full-resolution rows per frame
};

// Where and when a frame is rendered.
struct Frame {
    long long time = 0;        // layer time, time_scale units
    long long time_step = 1;   // one frame, time_scale units
    long long time_scale = 30;
    int full_w = 0, full_h = 0;      // layer size, full resolution
    int canvas_w = 0, canvas_h = 0;  // the whole layer at the current resolution
    double ds_x = 1, ds_y = 1;       // current resolution / full resolution
    int bits = 8;
    int in_w = 0, in_h = 0, in_ox = 0, in_oy = 0;      // input buffer and its corner on the canvas
    int out_w = 0, out_h = 0, out_ox = 0, out_oy = 0;  // output buffer and its corner on the canvas
};

// Microsoft C runtime rand(): a fixed, portable sequence (the kernels use the same one).
class LegacyRandom {
public:
    explicit LegacyRandom(uint32_t seed) : state_(seed) {}
    int next() {
        state_ = state_ * 214013u + 2531011u;
        return int((state_ >> 16) & 0x7fff);
    }
    float u() { return float(next() % 10000) / 10000.0f; }  // [0, 1), four decimals

private:
    uint32_t state_;
};

inline uint32_t mix32(uint32_t v) {
    v ^= v >> 16;
    v *= 0x7feb352du;
    v ^= v >> 15;
    v *= 0x846ca68bu;
    v ^= v >> 16;
    return v;
}

// Layer frame number of a time.
inline int frame_number(const Frame& f) {
    if (f.time_step <= 0) return 0;
    const long long n = f.time / f.time_step;
    return int(std::clamp<long long>(n, -100000000LL, 100000000LL));
}

// Distortion strength at frame `frame`: a value in [min, max] % that changes
// every Tempo Min..Tempo Max frames.
inline double distortion_strength(const Params& c, int frame) {
    double lo = c.dist_min, hi = c.dist_max, tlo = c.tempo_min, thi = c.tempo_max;
    if (lo > hi) std::swap(lo, hi);
    if (tlo > thi) std::swap(tlo, thi);
    LegacyRandom rng(uint32_t(c.seed));
    double d = 0;
    int count = -1;
    for (int i = 0; i <= std::max(0, frame); ++i) {
        if (count-- < 0) {
            d = (rng.u() * (hi - lo) + lo) / 100.0;
            count = int(rng.u() * (thi - tlo) + tlo);
        }
    }
    return d;
}

// Position of the bar band after `steps` steps of `step` that restart at -back
// once past `len` (closed form of the original's per-step loop).
inline double bar_position(long long steps, double step, double len, double back) {
    if (steps <= 0) return 0;
    const long long first = (long long)std::floor(len / step) + 1;  // steps until the first restart
    if (steps < first) return double(steps) * step;
    const long long period = (long long)std::floor((len + back) / step) + 1;
    return -back + double((steps - first) % period) * step;
}

// Vertical roll after `frame` frames, in full-resolution rows.
inline int roll_offset(const Params& c, int frame, int full_h) {
    if (c.roll_speed <= 0 || c.roll_prob <= 0 || full_h <= 0) return 0;
    LegacyRandom rng(uint32_t(c.seed));
    long long off = 0;
    int speed = 0, count = 0;
    for (int i = 0; i < frame; ++i) {
        off += speed;
        if (count-- < 0) {
            off = 0;
            if (rng.u() * 100.0f < c.roll_prob) {
                speed = int(rng.u() * c.roll_speed);
                count = int(rng.u() * c.roll_length);
            } else {
                speed = 0;
                count = int(rng.u() * (c.roll_length * 2.0 - c.roll_length / 2.0) + c.roll_length / 2.0);
            }
        }
    }
    return int(off % full_h);
}

inline ReceptionParams make_params(const Params& c, const Frame& f) {
    ReceptionParams p{};
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
    p.d_pitch = p.t_pitch = p.g_pitch = p.canvas_w;

    const int frame = frame_number(f);
    p.frame = frame;
    p.seed = uint32_t(c.seed);
    p.key = mix32(mix32(p.seed ^ 0x2c1b3c6du) ^ mix32(uint32_t(frame) * 0x9e3779b9u + 0x7f4a7c15u));
    p.edges = std::clamp(int(c.edges), 1, 4);
    p.blend = float(std::clamp(c.blend / 100.0, 0.0, 1.0));
    p.color_max = f.bits >= 32 ? 1e30f : 255.0f;

    // Color.
    p.tint_r = float(std::clamp(c.tint.r, 0.0, 1.0));
    p.tint_g = float(std::clamp(c.tint.g, 0.0, 1.0));
    p.tint_b = float(std::clamp(c.tint.b, 0.0, 1.0));
    p.lift = float(c.lift);
    p.bright_var = float(c.bright_var);
    p.contrast = float(c.contrast * 2.55);
    p.contrast_var = float(c.contrast_var);

    // Noise.
    p.noise_max = float(std::max(0.0, c.noise_max) * 2.55);
    p.noise_var = float(c.noise_var);
    p.noise_tempo = float(std::max(0.0, c.noise_tempo));
    p.micro = float(std::max(0.0, c.micro) * 2.55);
    p.noise_len = float(std::max(1.0, c.noise_width / 2.0));

    // Distortion.
    p.dist_on = (c.dist_min > 0 || c.dist_max > 0) ? 1 : 0;
    if (p.dist_on) {
        const double d = distortion_strength(c, frame);
        const double fr = double(frame);
        p.amp1 = float(std::fabs(std::cos(fr / std::max(1e-3, 100.0 - c.speed1) + 50.0)) * c.amp1 * d);
        p.amp2 = float(std::cos((fr + 50.0) / std::max(1e-3, 100.0 - c.speed2) + 20.0) * c.amp2 * d);
        p.ph1 = float(fr / 12.0 + 20.0);
        p.ph2 = float(10000.0 - (fr / 12.0 + 90.0));
        p.freq1 = float(c.freq1);
        p.freq2 = float(c.freq2);
        p.wave_div = float(p.full_h * 0.1 + 1.0);
        p.offset_var = float(c.offset_var);
    }

    // Glow: the tent radius is twice Glow Size, full-resolution pixels.
    const double radius = c.glow_size * 2.0;
    p.glow_on = (c.glow_size > 0.01 && radius >= 1.0 && c.glow_intensity != 0 && c.glow_threshold < 100) ? 1 : 0;
    p.glow_radius = std::clamp(int(std::lround(radius * p.ds_x)), 1, 128);
    p.glow_threshold = float(c.glow_threshold / 100.0);
    p.glow_intensity = float(c.glow_intensity);

    // Ghosting: three offsets that jitter from frame to frame.
    p.ghost_on = (c.w_ghost1 > 0 || c.w_ghost2 > 0 || c.w_ghost3 > 0) ? 1 : 0;
    if (p.ghost_on) {
        LegacyRandom rng(p.seed + uint32_t(frame));
        int o1 = 0, o2 = 0, o3 = 0;
        for (int i = 0; i < 4; ++i) {
            o1 = int(o1 + rng.u() * 2.0f - 1.0f);
            o2 = int(o2 + rng.u() * 10.0f - 5.0f);
            o3 = int(o3 + rng.u() * 20.0f - 10.0f);
            o1 = std::clamp(o1, -int(c.ghost_off1), int(c.ghost_off1));
            o2 = std::clamp(o2, -int(c.ghost_off2), int(c.ghost_off2));
            o3 = std::clamp(o3, -int(c.ghost_off3), int(c.ghost_off3));
        }
        p.ghost_off1 = int(std::lround(o1 * p.ds_x));
        p.ghost_off2 = int(std::lround(o2 * p.ds_x));
        p.ghost_off3 = int(std::lround(o3 * p.ds_x));
        p.w_primary = float(c.w_primary / 100.0);
        p.w_ghost1 = float(c.w_ghost1 / 100.0);
        p.w_ghost2 = float(c.w_ghost2 / 100.0);
        p.w_ghost3 = float(c.w_ghost3 / 100.0);
    }

    // Bars: a band of Bar Height % of the cycle, moving 5% of a frame height per frame.
    p.bars_on = (c.bar_dark > 0 || c.bar_height > 0) ? 1 : 0;
    if (p.bars_on) {
        const double step = std::max(0.0, c.bar_ripple) + 1e-4;
        const double back = p.full_h * step + 1e-6;
        const double len = back * c.bar_height / 100.0;
        const long long steps = (long long)std::ceil(double(frame) * p.full_h * 0.05);
        p.bar_step = float(step);
        p.bar_back = float(-back);
        p.bar_len = float(len);
        p.bar_pos = len > 0 ? float(bar_position(steps, step, len, back)) : 0.0f;
        p.bar_dark = float(c.bar_dark / 100.0);
        p.bar_phase = float(frame / 40.0);
    }

    // Vertical hold.
    const int roll = roll_offset(c, frame, p.full_h);
    p.roll = int(std::floor(roll * double(p.ds_y))) % p.canvas_h;
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

}  // namespace reception
