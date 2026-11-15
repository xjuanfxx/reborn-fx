// Tests for the Crosstalk renderer (CPU path; tools/gpu_check compares the GPU).
#include <cmath>
#include <cstdio>
#include <vector>

#include "crosstalk_cpu.h"

namespace {

using crosstalk::float4;

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

crosstalk::Frame frame(int w, int h, double ds = 1.0) {
    crosstalk::Frame f;
    f.full_w = w;
    f.full_h = h;
    f.canvas_w = f.in_w = f.out_w = int(w * ds);
    f.canvas_h = f.in_h = f.out_h = int(h * ds);
    f.ds_x = f.ds_y = ds;
    f.time = 0;
    f.time_scale = 30;
    return f;
}

// Settings that leave the picture alone (Process Input 100% of an untinted picture).
crosstalk::Params neutral() {
    crosstalk::Params c;
    for (crosstalk::Field* f : {&c.even, &c.odd}) {
        f->process = 100;
        f->tint = false;
        f->luminance = 0;
        f->noise = 0;
        f->decay = 1;
        f->comb = 0;
        f->hadj = 0;
    }
    return c;
}

std::vector<float4> render(const crosstalk::Params& c, const crosstalk::Frame& f, const std::vector<float4>& in) {
    std::vector<float4> out(size_t(f.out_w) * f.out_h);
    crosstalk::render_cpu(crosstalk::make_params(c, f), in.data(), out.data());
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
    const crosstalk::Frame f = frame(w, h);

    check(max_diff(render(neutral(), f, in), in) < 1e-5, "neutral settings keep the picture");

    {
        crosstalk::Params c;  // defaults
        c.blend = 100;
        check(max_diff(render(c, f, in), in) < 1e-6, "Blend With Original 100% returns the source");
    }

    {
        // Even bands of 2 rows black, odd bands of 3 rows untouched.
        crosstalk::Params c = neutral();
        c.even.size = 2;
        c.odd.size = 3;
        c.even.luminance = -100;
        const std::vector<float4> out = render(c, f, in);
        bool ok = true;
        for (int y = 0; y < 10; ++y) {
            const bool even = (y % 5) < 2;
            const float4 o = out[size_t(y) * w + 40], s = in[size_t(y) * w + 40];
            ok &= even ? (o.x == 0 && o.y == 0 && o.z == 0 && o.w == 1) : std::fabs(o.z - s.z) < 1e-6f;
        }
        check(ok, "bands are exactly Size rows (2 even, 3 odd)");
    }

    {
        // Process Input 100% with a white tint: grey = (R + G + B) / 3.
        crosstalk::Params c = neutral();
        c.even.tint = c.odd.tint = true;
        const std::vector<float4> out = render(c, f, in);
        const float4 s = in[size_t(8) * w + 10], o = out[size_t(8) * w + 10];  // row 8: even field
        const float l = (s.x + s.y + s.z) / 3;
        check(std::fabs(o.x - l * 248 / 255.0f) < 1e-5f && std::fabs(o.y - l) < 1e-5f,
              "tint maps luminance to the tint colour");
    }

    {
        // Defaults on a dark picture: odd rows keep it (the source comes back on
        // top of the clipped echo), even rows are darker but not black.
        std::vector<float4> dark(size_t(w) * h, float4{0.35f, 0.1f, 0.1f, 1.0f});
        crosstalk::Params c;
        c.odd.comb = 0;
        c.even.noise = c.odd.noise = 0;
        const std::vector<float4> out = render(c, f, dark);
        const float4 odd = out[size_t(2) * w + 50], even = out[size_t(0) * w + 50];
        check(odd.x > 0.3f && even.x > 0.05f && even.x < odd.x, "default look: dark picture survives");
    }

    {
        // Shift: -Horiz. Adj. % of the width (Influence 1000 = x1), Tile edges.
        crosstalk::Params c = neutral();
        c.even.method = c.odd.method = crosstalk::Method::Shift;
        c.even.hadj = c.odd.hadj = -10;  // 16 pixels to the left
        const std::vector<float4> out = render(c, f, in);
        bool ok = true;
        for (int x = 0; x < w; ++x) ok &= std::fabs(out[size_t(5) * w + x].z - in[size_t(5) * w + (x + 16) % w].z) < 1e-6f;
        check(ok, "Shift displaces rows by Horiz. Adj. % of the width (tiled)");
    }

    {
        crosstalk::Params c = neutral();
        c.even.method = c.odd.method = crosstalk::Method::WaveMotion;
        c.even.hadj = c.odd.hadj = 100;
        c.influence = 2;
        check(max_diff(render(c, f, in), in) > 0.01, "Wave Motion displaces rows");
    }

    {
        // Comb level 1: odd columns blank at full resolution, half coverage at half resolution.
        crosstalk::Params c = neutral();
        c.even.comb = c.odd.comb = 1;
        const std::vector<float4> full = render(c, f, in);
        check(full[1].w == 0 && full[2].w == 1, "comb blanks odd columns");
        const crosstalk::Frame half = frame(w, h, 0.5);
        const std::vector<float4> in_half = picture(half.in_w, half.in_h);
        const std::vector<float4> out_half = render(c, half, in_half);
        check(std::fabs(out_half[3].w - 0.5f) < 1e-6f, "comb keeps its average at half resolution");
    }

    {
        crosstalk::Params c;  // defaults: noise and echo
        const std::vector<float4> a = render(c, f, in), b = render(c, f, in);
        check(max_diff(a, b) == 0, "rendering is deterministic");
        crosstalk::Frame f2 = f;
        f2.time = 1;
        check(max_diff(a, render(c, f2, in)) > 0.01, "noise changes every frame");
        c.seed = 7;
        check(max_diff(a, render(c, f, in)) > 0.01, "noise changes with the seed");
    }

    {
        // Half resolution matches a 2x2 average of the full render (band and wave geometry).
        crosstalk::Params c = neutral();
        c.even.size = 4;
        c.odd.size = 4;
        c.even.luminance = -50;
        c.even.method = c.odd.method = crosstalk::Method::Wave;
        c.even.hadj = c.odd.hadj = 4;
        const std::vector<float4> full = render(c, f, in);
        const crosstalk::Frame half = frame(w, h, 0.5);
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
