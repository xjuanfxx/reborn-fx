// test_core — checks the CPU renderer without After Effects.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "analog_cpu.h"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("  %s  %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

using analog::float4;

// BGRA, premultiplied: a soft gradient with a few flat color patches.
std::vector<float4> test_image(int w, int h, float alpha = 1.0f) {
    std::vector<float4> img(size_t(w) * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float r = float(x) / w, g = float(y) / h, b = 0.5f;
            if (x > w / 4 && x < w / 2 && y > h / 4 && y < h / 2) r = 0.9f, g = 0.2f, b = 0.1f;
            if (x > w / 2 && x < 3 * w / 4 && y > h / 2 && y < 3 * h / 4) r = 0.1f, g = 0.3f, b = 0.8f;
            img[size_t(y) * w + x] = {b * alpha, g * alpha, r * alpha, alpha};
        }
    return img;
}

std::vector<float4> flat_image(int w, int h, float r, float g, float b) {
    return std::vector<float4>(size_t(w) * h, float4{b, g, r, 1.0f});
}

analog::Frame frame_for(int w, int h, double ds = 1.0, long long time = 0) {
    analog::Frame f;
    f.time = time;
    f.time_scale = 30;
    f.canvas_w = int(std::lround(w * ds));
    f.rows = int(std::lround(h * ds));
    f.ds_x = f.ds_y = ds;
    f.in_w = f.out_w = f.canvas_w;
    f.in_h = f.out_h = f.rows;
    return f;
}

std::vector<float4> render(const analog::Params& c, const analog::Frame& f, const std::vector<float4>& in) {
    std::vector<float4> out(size_t(f.out_w) * f.out_h);
    analog::render_cpu(analog::make_params(c, f), in.data(), out.data());
    return out;
}

// Everything that adds noise or the tape look switched off.
analog::Params clean() {
    analog::Params c;
    c.luma_noise = c.chroma_noise = c.hue_noise = 0;
    c.chroma_loss = 0;
    c.fm = false;
    c.vhs = analog::Vhs::Off;
    c.sharpen_on = false;
    c.amplification = 0;
    c.lp_in = false;
    c.lp_out = analog::LowPassOut::Off;
    c.scan_blur = 0;
    c.strength = 50;
    return c;
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

bool finite_in_range(const std::vector<float4>& img) {
    for (const float4& v : img) {
        for (float c : {v.x, v.y, v.z, v.w})
            if (!std::isfinite(c) || c < 0.0f || c > 1.0f) return false;
    }
    return true;
}

}  // namespace

int main() {
    const int w = 320, h = 180;
    const std::vector<float4> img = test_image(w, h);

    std::printf("defaults\n");
    {
        const analog::Params c;
        const analog::Frame f = frame_for(w, h, 1.0, 7);
        const auto a = render(c, f, img), b = render(c, f, img);
        check(max_diff(a, b) == 0.0, "same frame renders identically (stateless)");
        check(finite_in_range(a), "output finite and within 0..1 at 8 bpc");
        const auto next = render(c, frame_for(w, h, 1.0, 8), img);
        check(max_diff(a, next) > 0.01, "noise changes from frame to frame");
        analog::Params other = c;
        other.seed = 5;
        check(max_diff(a, render(other, f, img)) > 0.01, "seed changes the noise");
    }

    std::printf("composite round trip\n");
    {
        const analog::Params c = clean();
        const auto flat = flat_image(w, h, 0.8f, 0.3f, 0.2f);
        const auto out = render(c, frame_for(w, h), flat);
        double m = 0;
        for (int y = 0; y < h; ++y)
            for (int x = 8; x < w - 8; ++x) {
                const float4& v = out[size_t(y) * w + x];
                m = std::max({m, double(std::fabs(v.z - 0.8f)), double(std::fabs(v.y - 0.3f)),
                              double(std::fabs(v.x - 0.2f))});
            }
        check(m < 0.01, "a flat color survives encode/decode (max diff " + std::to_string(m) + ")");

        analog::Params mono = c;
        mono.strength = 0;
        const auto grey = render(mono, frame_for(w, h), flat);
        const float4& v = grey[size_t(h / 2) * w + w / 2];
        check(std::fabs(v.x - v.z) < 1e-4 && std::fabs(v.y - v.z) < 1e-4, "no subcarrier: black and white");
    }

    std::printf("preview resolution\n");
    {
        analog::Params c = clean();
        c.vhs = analog::Vhs::SP;
        c.sharpen_on = true;
        c.fm = true;
        c.tape_errors = false;
        const auto full = render(c, frame_for(w, h), img);
        // Half resolution input: box-downsampled full input.
        const int hw = w / 2, hh = h / 2;
        std::vector<float4> half_in(size_t(hw) * hh), full_ds(size_t(hw) * hh);
        auto down = [&](const std::vector<float4>& src, std::vector<float4>& dst) {
            for (int y = 0; y < hh; ++y)
                for (int x = 0; x < hw; ++x) {
                    float4 s{0, 0, 0, 0};
                    for (int dy = 0; dy < 2; ++dy)
                        for (int dx = 0; dx < 2; ++dx) {
                            const float4& v = src[size_t(2 * y + dy) * w + 2 * x + dx];
                            s.x += v.x / 4; s.y += v.y / 4; s.z += v.z / 4; s.w += v.w / 4;
                        }
                    dst[size_t(y) * hw + x] = s;
                }
        };
        down(img, half_in);
        down(full, full_ds);
        const auto half = render(c, frame_for(w, h, 0.5), half_in);
        double mean = 0;
        for (size_t i = 0; i < half.size(); ++i)
            mean += std::fabs(half[i].x - full_ds[i].x) + std::fabs(half[i].y - full_ds[i].y) +
                    std::fabs(half[i].z - full_ds[i].z);
        mean /= double(half.size()) * 3;
        check(mean < 0.03, "half resolution looks like the full render (mean diff " + std::to_string(mean) + ")");
    }

    std::printf("layout\n");
    {
        const analog::Params c;
        const analog::Frame f = frame_for(w, h, 1.0, 3);
        const auto full = render(c, f, img);
        analog::Frame sub = f;
        sub.out_ox = 17;
        sub.out_oy = 9;
        sub.out_w = 100;
        sub.out_h = 60;
        const auto part = render(c, sub, img);
        double m = 0;
        for (int y = 0; y < sub.out_h; ++y)
            for (int x = 0; x < sub.out_w; ++x) {
                const float4& a = part[size_t(y) * sub.out_w + x];
                const float4& b = full[size_t(y + sub.out_oy) * w + x + sub.out_ox];
                m = std::max({m, double(std::fabs(a.x - b.x)), double(std::fabs(a.y - b.y)),
                              double(std::fabs(a.z - b.z)), double(std::fabs(a.w - b.w))});
            }
        check(m < 1e-6, "an output sub-rect equals the same part of the full frame");

        // Input smaller than the layer and offset: outside is transparent.
        analog::Frame off = f;
        off.in_ox = 20;
        off.in_oy = 10;
        off.in_w = w - 40;
        off.in_h = h - 20;
        std::vector<float4> in2(size_t(off.in_w) * off.in_h);
        for (int y = 0; y < off.in_h; ++y)
            for (int x = 0; x < off.in_w; ++x) in2[size_t(y) * off.in_w + x] = img[size_t(y + 10) * w + x + 20];
        const auto o2 = render(c, off, in2);
        check(o2[0].w == 0.0f && o2[size_t(h / 2) * w + w / 2].w == 1.0f, "alpha follows the input placement");
        check(finite_in_range(o2), "offset input stays finite");
    }

    std::printf("controls\n");
    {
        analog::Params c = clean();
        c.roll_speed = 30;  // 30 px per second, 1 s in: 30 rows
        const auto rolled = render(c, frame_for(w, h, 1.0, 30), img);
        const auto still = render(clean(), frame_for(w, h, 1.0, 30), img);
        double m = 0;
        for (int y = 31; y < h; ++y)  // row 30's vertical chroma blend sees the wrapped bottom row
            for (int x = 0; x < w; ++x) {
                const float4& a = rolled[size_t(y) * w + x];
                const float4& b = still[size_t(y - 30) * w + x];
                m = std::max({m, double(std::fabs(a.z - b.z)), double(std::fabs(a.y - b.y))});
            }
        check(m < 1e-5, "Roll Speed slides the picture down");

        analog::Params d = c;
        d.roll_speed = 0;
        d.destruction = true;
        d.strength = 200;
        const auto boom = render(d, frame_for(w, h), img);
        check(finite_in_range(boom), "Destruction wraps instead of overflowing");

        analog::Params e;
        e.thick = 30;
        e.fine = 5;
        e.dist_speed = 20;
        e.generations = 3;
        e.svideo = false;
        e.vhs = analog::Vhs::EP;
        e.fm_quantize = 6;
        e.lp_out = analog::LowPassOut::Lite;
        e.phase_shift = 180;
        e.phase_offset = 1;
        e.scan_blur = 3;
        const auto heavy = render(e, frame_for(w, h, 1.0, 11), img);
        check(finite_in_range(heavy), "heavy settings stay finite");
        check(max_diff(heavy, render(e, frame_for(w, h, 1.0, 11), img)) == 0.0, "heavy settings deterministic");

        // 32 bpc keeps values above 1.
        analog::Params hot = clean();
        auto bright = flat_image(w, h, 2.0f, 2.0f, 2.0f);
        analog::Frame f32 = frame_for(w, h);
        f32.bits = 32;
        const auto o32 = render(hot, f32, bright);
        check(o32[size_t(h / 2) * w + w / 2].z > 1.5f, "32 bpc keeps over-range values");
    }

    std::printf(failures ? "%d check(s) failed\n" : "All core checks passed\n", failures);
    return failures ? 1 : 0;
}
