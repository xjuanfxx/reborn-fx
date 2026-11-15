// analog_core.h — the effect's controls and how they become an AnalogParams block.
//
// Independent of After Effects. The plugin reads the controls into Params,
// make_params() adds the frame (time, size, resolution, bit depth) and the
// renderers (analog_cpu.h, plugin/gpu) run the passes in analog_kernels.inc.
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

#include "analog_types.h"

namespace analog {

enum class Vhs { Off = 1, SP = 2, LP = 3, EP = 4 };
enum class LowPassOut { Off = 1, Lite = 2, Strong = 3 };

// Defaults follow the classic layout of this kind of effect (docs/behaviour.md).
struct Params {
    double seed = 0;
    double strength = 40;        // Signal Strength: chroma subcarrier amplitude x50
    double amplification = 1;    // Signal Amplification: pre-emphasis amount
    double cutoff = 88;          // Cutoff Filter: pre-emphasis corner = 315 MHz / value
    double luma_noise = 1;
    double chroma_noise = 1;
    double hue_noise = 15;
    double chroma_loss = 0.24;   // probability per line
    double sharpen = 2;
    double generations = 0;      // Device-to-Device Connections
    Vhs vhs = Vhs::SP;
    bool destruction = false;

    bool fm = true;              // Luma Modulation
    double fm_freq = 0.62;       // Frequency (carrier, radians per sample)
    double fm_dev = 2;           // Phase (deviation, radians per sample)
    double fm_light = 0.62;      // Lightness (detector level)
    double fm_quantize = 0;
    bool tape_errors = true;
    double te_start = 0.92, te_end = 1, te_prob = 0.1, te_fade = 5, te_gamma = 3;

    double thick = 0;            // TV Distortion
    double fine = 0;
    double dist_speed = 0;
    double roll_speed = 0;       // full-resolution pixels per second

    bool lp_in = true;           // Experimental
    LowPassOut lp_out = LowPassOut::Strong;
    int phase_shift = 0;         // 0, 90, 180 or 270 degrees
    int phase_offset = 0;        // 0..3
    bool svideo = true;
    bool vblend = true;
    bool sharpen_on = true;
    double scan_blur = 1.5;
};

// Where and when a frame is rendered.
struct Frame {
    long long time = 0;  // layer time, time_scale units
    long long time_scale = 30;
    int canvas_w = 0, rows = 0;   // canvas (the whole layer) at the current resolution
    double ds_x = 1, ds_y = 1;    // current resolution / full resolution
    int bits = 8;
    int in_w = 0, in_h = 0, in_ox = 0, in_oy = 0;   // input buffer and its corner on the canvas
    int out_w = 0, out_h = 0, out_ox = 0, out_oy = 0;  // output buffer and its corner on the canvas
};

constexpr double kSampleRate = 14318181.818181818;  // four samples per NTSC subcarrier cycle

// One-pole low-pass coefficient for corner frequency fc at sample rate rate.
inline float one_pole(double rate, double fc) {
    if (fc <= 0 || rate <= 0) return 0.0f;
    return float((1.0 / rate) / (1.0 / (2.0 * 3.141592653589793 * fc) + 1.0 / rate));
}

inline uint32_t mix32(uint32_t v) {
    v ^= v >> 16;
    v *= 0x7feb352du;
    v ^= v >> 15;
    v *= 0x846ca68bu;
    v ^= v >> 16;
    return v;
}

inline AnalogParams make_params(const Params& c, const Frame& f) {
    AnalogParams p{};
    p.canvas_w = std::max(1, f.canvas_w);
    p.rows = std::max(1, f.rows);
    p.ds_x = float(std::clamp(f.ds_x, 1e-3, 1.0));
    p.ds_y = float(std::clamp(f.ds_y, 1e-3, 1.0));
    p.inv_ds_x = 1.0f / p.ds_x;
    p.inv_ds_y = 1.0f / p.ds_y;
    p.line_w = std::max(1, int(std::lround(p.canvas_w / double(p.ds_x))));
    p.full_h = std::max(1, int(std::lround(p.rows / double(p.ds_y))));
    p.plane_sx = 1;
    p.plane_sy = p.line_w;
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

    const double seconds = f.time_scale > 0 ? double(f.time) / double(f.time_scale) : 0.0;
    p.time_s = float(seconds);
    const uint32_t seed = uint32_t(int64_t(std::floor(c.seed)));
    p.seed_key = mix32(seed ^ 0x51ed270bu);
    p.key = mix32(p.seed_key ^ mix32(uint32_t(f.time) ^ mix32(uint32_t(f.time >> 32) + 0x2545f491u) ^
                                     mix32(uint32_t(f.time_scale) * 0x9e3779b9u)));

    // Roll: the picture slides down by roll_speed full-resolution pixels per second.
    if (c.roll_speed > 0) {
        const double off = std::fmod(seconds * c.roll_speed, double(p.full_h));
        int r = int(std::floor((off < 0 ? off + p.full_h : off) * p.ds_y));
        p.roll = ((r % p.rows) + p.rows) % p.rows;
    }

    // Composite signal.
    p.lp_in = c.lp_in ? 1 : 0;
    p.lp_out = c.lp_out == LowPassOut::Lite ? 1 : (c.lp_out == LowPassOut::Strong ? 2 : 0);
    p.phase_mode = std::clamp(c.phase_shift / 90, 0, 3);
    p.phase_offset = c.phase_offset & 3;
    p.amp = float(std::max(0.0, c.strength) / 50.0);
    p.inv_amp = p.amp > 0 ? 1.0f / p.amp : 0.0f;
    p.pre_amount = float(c.amplification);
    p.a_pre = c.cutoff > 0 ? one_pole(kSampleRate, 315000000.0 / c.cutoff) : 0.0f;
    p.a_lpin_i = one_pole(kSampleRate, 1300000.0);
    p.a_lpin_q = one_pole(kSampleRate, 600000.0);
    p.a_lpout_lite = one_pole(kSampleRate, 2600000.0);
    // The original works on 8-bit values x256; signal units are 8-bit values.
    p.noise_luma = float(std::max(0.0, c.luma_noise) * 1000.0 / 256.0);
    p.noise_chroma = float(std::max(0.0, c.chroma_noise) * 3750.0 / 256.0);
    p.hue_noise = float(std::max(0.0, c.hue_noise));
    p.chroma_loss = float(std::clamp(c.chroma_loss, 0.0, 1.0));
    p.generations = std::clamp(int(c.generations), 0, 100);
    p.svideo = c.svideo ? 1 : 0;

    // Tape.
    double luma_cut = 2400000, chroma_cut = 320000;
    int delay = 9;
    switch (c.vhs) {
        case Vhs::LP: luma_cut = 1900000; chroma_cut = 300000; delay = 12; break;
        case Vhs::EP: luma_cut = 1400000; chroma_cut = 280000; delay = 14; break;
        default: break;
    }
    p.vhs = c.vhs == Vhs::Off ? 0 : int(c.vhs) - 1;
    p.chroma_delay = delay;
    p.a_luma = one_pole(kSampleRate, luma_cut);
    p.a_luma_slow = one_pole(kSampleRate, std::floor(luma_cut / 3));
    p.a_chroma = one_pole(kSampleRate, chroma_cut);
    p.vblend = c.vblend ? 1 : 0;
    p.sharpen = c.sharpen_on ? 1 : 0;
    p.sharpen_amount = float(c.sharpen);
    p.a_sharpen = one_pole(kSampleRate, 4 * luma_cut);

    // FM luma. The detector low-pass uses the tape's luma/chroma corners.
    p.fm = c.fm && c.fm_dev > 0 ? 1 : 0;
    p.fm_quantize = std::max(0, int(c.fm_quantize));
    p.fm_freq = float(c.fm_freq);
    p.fm_dev = float(std::max(1e-3, c.fm_dev));
    p.fm_light = float(c.fm_light);
    p.a_fm = one_pole(luma_cut, chroma_cut);
    p.tape_errors = c.tape_errors ? 1 : 0;
    p.te_y0 = float(std::floor(c.te_start * p.full_h));
    p.te_y1 = float(std::floor(c.te_end * p.full_h));
    p.te_prob = float(c.te_prob);
    p.te_fade = float(c.te_fade);
    p.te_gamma = float(c.te_gamma);

    // TV distortion, in full-resolution samples.
    p.dist_thick = float(std::max(0.0, c.thick) / 100.0 * p.line_w);
    p.dist_fine = float(std::max(0.0, c.fine) / 10.0 * 2.5 / 100.0 * p.line_w);
    p.dist_rate = float(std::max(0.0, c.dist_speed) / 10.0);

    // Output.
    p.destruction = c.destruction ? 1 : 0;
    p.color_max = f.bits >= 32 ? 1e30f : 1.0f;
    if (c.scan_blur > 1.0) {
        p.blur_w = std::max(1, int(p.line_w / c.scan_blur));
        p.blur_h = std::max(1, int(p.full_h / c.scan_blur));
        if (p.blur_w < p.canvas_w || p.blur_h < p.rows) {
            p.blur = float(c.scan_blur);
            p.blur_w = std::min(p.blur_w, p.canvas_w);
            p.blur_h = std::min(p.blur_h, p.rows);
        }
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

}  // namespace analog
