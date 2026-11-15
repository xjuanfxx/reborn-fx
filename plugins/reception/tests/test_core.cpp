// Tests for the Reception renderer (CPU path; tools/gpu_check compares the GPU).
#include <cmath>
#include <cstdio>
#include <vector>

#include "reception_cpu.h"

namespace {

using reception::float4;

int failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

// Opaque, smooth test picture (B, G, R, A).
std::vector<float4> picture(int w, int h) {
    std::vector<float4> img(size_t(w) * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            img[size_t(y) * w + x] = {0.2f + 0.6f * float(x) / w, 0.3f + 0.4f * float(y) / h,
                                      0.5f + 0.3f * std::sin(float(x + y) * 0.05f), 1.0f};
    return img;
}

reception::Frame frame(int w, int h, int number = 0, double ds = 1.0) {
    reception::Frame f;
    f.full_w = w;
    f.full_h = h;
    f.canvas_w = f.in_w = f.out_w = int(w * ds);
    f.canvas_h = f.in_h = f.out_h = int(h * ds);
    f.ds_x = f.ds_y = ds;
    f.time = number;
    f.time_step = 1;
    f.time_scale = 30;
    return f;
}

// Every stage off and a white tint: the picture passes through.
reception::Params neutral() {
    reception::Params c;
    c.tint = {1, 1, 1};
    c.bright_var = 0;
    c.noise_max = c.micro = 0;
    c.dist_min = c.dist_max = 0;
    c.glow_size = 0;
    c.w_ghost1 = c.w_ghost2 = c.w_ghost3 = 0;
    c.bar_dark = c.bar_height = 0;
    c.roll_prob = 0;
    return c;
}

std::vector<float4> render(const reception::Params& c, const reception::Frame& f, const std::vector<float4>& in) {
    std::vector<float4> out(size_t(f.out_w) * f.out_h);
    reception::render_cpu(reception::make_params(c, f), in.data(), out.data());
    return out;
}

double max_diff(const std::vector<float4>& a, const std::vector<float4>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, double(std::fabs(a[i].x - b[i].x)));
        m = std::max(m, double(std::fabs(a[i].y - b[i].y)));
        m = std::max(m, double(std::fabs(a[i].z - b[i].z)));
        m = std::max(m, double(std::fabs(a[i].w - b[i].w)));
    }
    return m;
}

}  // namespace

int main() {
    const int w = 160, h = 90;
    const std::vector<float4> in = picture(w, h);
    const reception::Frame f = frame(w, h, 12);

    check(max_diff(render(neutral(), f, in), in) < 1e-5, "neutral settings keep the picture");

    {
        reception::Params c;  // defaults
        c.blend = 100;
        check(max_diff(render(c, f, in), in) < 1e-6, "Blend With Original 100% returns the source");
    }

    {
        // The contrast curve: identity at 0, S-shaped above, flattened below, symmetric around 127.5.
        using reception::kernels::rc_contrast;
        bool ok = rc_contrast(40, 0) == 40 && rc_contrast(40, 60) < 40 && rc_contrast(215, 60) > 215 &&
                  rc_contrast(40, -60) > 40 && std::fabs(rc_contrast(40, 60) + rc_contrast(215, 60) - 255) < 1e-3;
        check(ok, "contrast curve");
    }

    {
        // Ghosts: primary 100%, ghost 3 weight 0.5 at a jittered offset; Tile edges.
        reception::Params c = neutral();
        c.w_primary = 100;
        c.w_ghost3 = 50;
        const ReceptionParams p = reception::make_params(c, f);
        const std::vector<float4> out = render(c, f, in);
        const int x = 70, y = 40, o = p.ghost_off3;
        const float want = std::fmin(1.0f, in[size_t(y) * w + x].z + 0.5f * in[size_t(y) * w + ((x + o) % w + w) % w].z);
        check(o != 0 && std::fabs(out[size_t(y) * w + x].z - want) < 1e-5f, "ghost adds a weighted, offset copy");
    }

    {
        // Vertical hold: a rolling frame is the still frame moved up by `roll` rows.
        reception::Params c = neutral();
        c.roll_prob = 100;
        c.roll_speed = 7;
        c.roll_length = 1000;
        int roll = 0, number = 0;
        for (c.seed = 0; c.seed < 50 && roll == 0; ++c.seed)
            for (number = 1; number < 50 && roll == 0; ++number) roll = reception::make_params(c, frame(w, h, number)).roll;
        --c.seed;
        const std::vector<float4> out = render(c, frame(w, h, number - 1), in);
        std::printf("      seed %d, frame %d: roll %d rows\n", c.seed, number - 1, roll);
        bool ok = roll != 0;
        for (int y = 0; y < h && ok; ++y) ok = std::fabs(out[size_t(y) * w + 30].z - in[size_t((y + roll) % h) * w + 30].z) < 1e-5f;
        check(ok, "vertical hold rolls the picture");
    }

    {
        // Distortion strength changes over time (Tempo) and stays within Min..Max.
        reception::Params c;
        c.dist_min = 20;
        c.dist_max = 90;
        c.tempo_min = c.tempo_max = 3;
        bool varies = false, inside = true;
        const double d0 = reception::distortion_strength(c, 0);
        for (int n = 0; n < 40; ++n) {
            const double d = reception::distortion_strength(c, n);
            varies |= d != d0;
            inside &= d >= 0.2 && d <= 0.9;
        }
        check(varies && inside, "distortion strength follows Min/Max and Tempo");
    }

    {
        // The closed-form bar position equals the original's step loop.
        bool ok = true;
        for (long long steps : {0LL, 5LL, 123LL, 4567LL, 98765LL}) {
            const double step = 3.0001, back = 90 * step + 1e-6, len = back * 0.4;
            double pos = 0;
            for (long long i = 0; i < steps; ++i) {
                pos += step;
                if (pos > len) pos = -back;
            }
            ok &= std::fabs(pos - reception::bar_position(steps, step, len, back)) < 1e-3 * (1 + steps / 1000.0);
        }
        check(ok, "bar position (closed form)");
    }

    {
        reception::Params c;  // defaults: noise, walks, ghosts, bars
        const std::vector<float4> a = render(c, f, in), b = render(c, f, in);
        check(max_diff(a, b) == 0, "rendering is deterministic");
        check(max_diff(a, render(c, frame(w, h, 13), in)) > 0.01, "noise changes every frame");
        c.seed = 9;
        check(max_diff(a, render(c, f, in)) > 0.01, "noise changes with the seed");
    }

    {
        // Half resolution matches a 2x2 average of the full render (rows, waves, bars, roll).
        reception::Params c = neutral();
        c.bright_var = 30;
        c.bar_dark = 60;
        c.bar_height = 50;
        c.amp1 = 20;
        c.dist_min = c.dist_max = 100;
        c.offset_var = 0;
        const reception::Frame big = frame(w, h, 33);
        const std::vector<float4> full = render(c, big, in);
        const reception::Frame half = frame(w, h, 33, 0.5);
        std::vector<float4> in_half(size_t(half.in_w) * half.in_h);
        for (int y = 0; y < half.in_h; ++y)
            for (int x = 0; x < half.in_w; ++x) {
                const float4 *p = &in[size_t(2 * y) * w + 2 * x], *q = p + w;
                in_half[size_t(y) * half.in_w + x] = {(p[0].x + p[1].x + q[0].x + q[1].x) / 4,
                                                      (p[0].y + p[1].y + q[0].y + q[1].y) / 4,
                                                      (p[0].z + p[1].z + q[0].z + q[1].z) / 4, 1};
            }
        const std::vector<float4> out_half = render(c, half, in_half);
        double err = 0;
        for (int y = 0; y < half.out_h; ++y)
            for (int x = 0; x < half.out_w; ++x) {
                const float4 *p = &full[size_t(2 * y) * w + 2 * x], *q = p + w;
                const float avg = (p[0].z + p[1].z + q[0].z + q[1].z) / 4;
                err += std::fabs(avg - out_half[size_t(y) * half.out_w + x].z);
            }
        err /= double(half.out_w) * half.out_h;
        std::printf("      mean full/half difference %.4f\n", err);
        check(err < 0.03, "half resolution looks like the full render");
    }

    std::printf(failures ? "%d test(s) failed\n" : "All tests passed\n", failures);
    return failures ? 1 : 0;
}
