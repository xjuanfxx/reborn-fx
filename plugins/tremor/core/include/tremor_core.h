// tremor_core.h — Tremor's motion model: parameters → per-frame state.
//
// Independent of After Effects. Every value is a pure function of the
// parameters and the time, so frames can render in any order and in parallel.
#pragma once

#include <cmath>
#include <cstdint>

namespace tremor {

enum class Deinterlace { None = 1, LowerOnly = 2, UpperOnly = 3, Merge = 4 };

// What shows where the moved picture leaves the frame.
enum class Edges { Transparent = 1, Mirror = 2, Tile = 3, Black = 4 };

// Parameter values as shown in the Effect Controls panel. Defaults match the panel.
struct Params {
    double blend = 0;  // Blend With Original, %: cross-fade back to the source
    int seed = 0;
    Deinterlace deinterlace = Deinterlace::None;
    Edges edges = Edges::Transparent;

    bool alpha_composite = false;
    double r_adjust = 0, g_adjust = 0, b_adjust = 0;  // %, -100..100

    double h_max = 16, h_freq = 2, h_amp = 0.1, h_phase = 30;
    double v_max = 4, v_freq = 2, v_amp = 0.1, v_phase = 30;
    double h_erratic = 0.6, v_erratic = 0.3;  // % chance per frame of a random jump

    double zoom_base = 0.1, zoom_factor = 0.1, zoom_freq = 7, zoom_amp = 0.1, zoom_phase = 90;
    double zoom_erratic = 2;

    double rot_max = 2, rot_freq = 16, rot_amp = 1, rot_phase = 15;
    double ch_sep = 4, ch_freq = 18, ch_amp = 0.4, ch_phase = 30;

    double erratic_h_adj = 100, erratic_v_adj = 100, erratic_zoom_adj = 100;  // %
};

// Everything the renderer needs for one frame. Lengths are in pixels of the
// buffer being rendered (already downsampled).
struct FrameState {
    double zoom = 0;            // magnification is 1 + zoom, around the center
    double rotation_deg = 0;    // red channel
    double separation_deg = 0;  // green = rotation + separation, blue = rotation + 2·separation
    double shift_x = 0, shift_y = 0;
    double center_x = 0, center_y = 0;
};

// Microsoft C runtime rand(): a fixed, portable sequence, so the same seed gives
// the same jumps on every platform.
class LegacyRandom {
public:
    explicit LegacyRandom(uint32_t seed) : state_(seed) {}
    int next() {
        state_ = state_ * 214013u + 2531011u;
        return int((state_ >> 16) & 0x7fff);
    }
    // A roll in [0, 1) with three decimals.
    double roll() { return double(next() % 1000) / 1000.0; }

private:
    uint32_t state_;
};

constexpr double kPi = 3.14159265358979323846;

// Phases are in degrees.
inline double wave(double freq, double phase_deg, double omega) {
    return std::cos(freq * omega + phase_deg * (kPi / 180.0));
}

// `time` and `time_scale`: the layer time as After Effects gives it (time / time_scale
// seconds); the raw `time` also seeds the jumps. `width`/`height`: buffer size.
inline FrameState evaluate(const Params& p, int32_t time, uint32_t time_scale, int width, int height) {
    FrameState s;
    const double omega = (time_scale ? double(time) / double(time_scale) : 0.0) * 2.0 * kPi;

    s.zoom = wave(p.zoom_freq, p.zoom_phase, omega) * (p.zoom_factor * p.zoom_amp * 0.1) + p.zoom_base;
    s.rotation_deg = wave(p.rot_freq, p.rot_phase, omega) * p.rot_amp * p.rot_max;
    s.separation_deg = wave(p.ch_freq, p.ch_phase, omega) * p.ch_amp * p.ch_sep;
    s.shift_y = wave(p.v_freq, p.v_phase, omega) * p.v_amp * (p.v_max * 0.01 * height);
    s.shift_x = wave(p.h_freq, p.h_phase, omega) * p.h_amp * (p.h_max * 0.01 * width);

    // Erratic jumps replace the smooth value on a few random frames.
    LegacyRandom rng(uint32_t(p.seed) + uint32_t(time));
    if (rng.roll() < p.v_erratic * 0.01)
        s.shift_y = (double(rng.next() % 1000) / 500.0 - 1.0) * height * (p.erratic_v_adj / 100.0);
    if (rng.roll() < p.h_erratic * 0.01)
        s.shift_x = (double(rng.next() % 1000) / 500.0 - 1.0) * width * (p.erratic_h_adj / 100.0);
    if (rng.roll() < p.zoom_erratic * 0.01)
        s.zoom = (double(rng.next() % 1000) / 100.0) * s.zoom * (p.erratic_zoom_adj / 100.0);

    // Center of the pixel grid (pixel centers sit at integer coordinates).
    s.center_x = (width - 1) * 0.5;
    s.center_y = (height - 1) * 0.5;
    return s;
}

}  // namespace tremor
