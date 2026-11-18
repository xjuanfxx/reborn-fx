// smear_bench — quality and speed benchmark for Smear, without After Effects.
//
// Renders synthetic scenes with known motion (procedurally textured layers
// moving and turning over differently moving backgrounds), so both the true
// optical flow and the true motion blur are known:
//
//   * frames −1, 0, +1 are rendered at their instants (4×4 supersampled);
//   * ground truth = 64 subframes spread over the shutter interval centred on
//     frame 0 (each with its own sub-pixel offset, so it is anti-aliased too).
//
// Per scene it reports the flow end-point error against the true flow, the
// PSNR and mean error of the blur against the ground truth, and a ghosting
// metric: the mean error restricted to the occlusion band (pixels whose
// visible layer changes between frames −1 and +1, where occlusion handling
// matters). Three renderers are compared:
//
//   naive      : blur along our flow, no occlusion handling (the baseline)
//   smear      : Smear with our flow
//   smear/true : Smear's reconstruction fed with the true flow (isolates the
//                reconstruction from flow errors)
//
// Then it times the full CPU pipeline (two flows + blur) at 1920×1080 and
// writes a few images as PPM.
//
// Usage: smear_bench [out_dir] [options]
//   --scene NAME        only this scene
//   --quality 1|2|3     Draft / Normal (default) / Best
//   --sensitivity S     Motion Sensitivity, 0–100 (default 50)
//   --no-images         do not write PPM images
//   --no-timing         skip the 1920×1080 timing
//   --true-flow         feed the true flow to every renderer (isolates reconstruction)
//   --debug             also report motion-vector error and layer classification,
//                       and write occlusion / layer / vector-error images
//   --probe X Y         with --debug: print the vectors and scores of pixel (X, Y)
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "smear_render.h"

namespace {

using smear::FlowField;
using smear::Image;
using smear::Pixel;
using smear::Vec2;

constexpr double kPi = 3.14159265358979323846;

// --- Procedural textures ------------------------------------------------------------

uint32_t hash3(int x, int y, uint32_t seed) {
    uint32_t h = uint32_t(x) * 0x8da6b343u ^ uint32_t(y) * 0xd8163841u ^ seed * 0xcb1ab31fu;
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    return h;
}

double lattice(int x, int y, uint32_t seed) { return double(hash3(x, y, seed) & 0xffffff) / double(0xffffff); }

double value_noise(double x, double y, uint32_t seed) {
    const double fx0 = std::floor(x), fy0 = std::floor(y);
    const int ix = int(fx0), iy = int(fy0);
    double fx = x - fx0, fy = y - fy0;
    fx = fx * fx * (3 - 2 * fx);
    fy = fy * fy * (3 - 2 * fy);
    const double a = lattice(ix, iy, seed), b = lattice(ix + 1, iy, seed);
    const double c = lattice(ix, iy + 1, seed), d = lattice(ix + 1, iy + 1, seed);
    return (a + (b - a) * fx) + ((c + (d - c) * fx) - (a + (b - a) * fx)) * fy;
}

double fbm(double x, double y, double scale, uint32_t seed) {
    double v = 0, amp = 0.5, total = 0, s = 1.0 / scale;
    for (int o = 0; o < 4; ++o) {
        v += amp * value_noise(x * s, y * s, seed + uint32_t(o) * 101u);
        total += amp;
        amp *= 0.55;
        s *= 2.0;
    }
    return v / total;
}

// Texture `kind` at local coordinates (u, v).
Pixel texture(int kind, double u, double v, uint32_t seed) {
    auto contrast = [](double t) { return std::clamp(0.5 + (t - 0.5) * 2.2, 0.0, 1.0); };
    double r = contrast(fbm(u, v, 22, seed)), g = contrast(fbm(u, v, 22, seed + 7)), b = contrast(fbm(u, v, 22, seed + 13));
    if (kind == 1) {  // foreground: bold stripes over noise, easy to spot as ghosts
        const double s = 0.5 + 0.5 * std::sin(u * 2 * kPi / 15.0) * std::cos(v * 2 * kPi / 41.0);
        r = 0.55 * r + 0.45 * s;
        g = 0.55 * g + 0.45 * (1 - s) * 0.8;
        b = 0.55 * b + 0.25 * s;
    }
    return {float(r), float(g), float(b), 1.0f};
}

// --- Scenes -----------------------------------------------------------------------

struct Object {
    bool background = false;  // fills the frame
    double cx = 0, cy = 0;    // centre at frame 0
    double vx = 0, vy = 0;    // pixels per frame
    double angle = 0, spin = 0;  // radians, radians per frame
    double hw = 0, hh = 0, corner = 0;  // half width/height, corner radius
    int tex = 0;
    uint32_t seed = 1;

    void pose(double t, double& x, double& y, double& a) const {
        x = cx + vx * t;
        y = cy + vy * t;
        a = angle + spin * t;
    }
    // Object coordinates of image point (px, py) at time t.
    void local(double px, double py, double t, double& u, double& v) const {
        double x, y, a;
        pose(t, x, y, a);
        const double c = std::cos(a), s = std::sin(a), dx = px - x, dy = py - y;
        u = c * dx + s * dy;
        v = -s * dx + c * dy;
    }
    bool contains(double u, double v) const {
        if (background) return true;
        const double qx = std::fabs(u) - (hw - corner), qy = std::fabs(v) - (hh - corner);
        const double ox = std::max(qx, 0.0), oy = std::max(qy, 0.0);
        return std::sqrt(ox * ox + oy * oy) + std::min(std::max(qx, qy), 0.0) <= corner;
    }
    // Image position at time t of the object point (u, v).
    void place(double u, double v, double t, double& px, double& py) const {
        double x, y, a;
        pose(t, x, y, a);
        const double c = std::cos(a), s = std::sin(a);
        px = x + c * u - s * v;
        py = y + s * u + c * v;
    }
};

struct Scene {
    std::string name, what;
    int width = 640, height = 360;
    float grain = 0;  // film grain added to every rendered frame (and to the truth), std. deviation
    std::vector<Object> objects;  // back to front

    int visible(double px, double py, double t, double& u, double& v) const {
        for (int i = int(objects.size()) - 1; i >= 0; --i) {
            objects[size_t(i)].local(px, py, t, u, v);
            if (objects[size_t(i)].contains(u, v)) return i;
        }
        return -1;
    }
    Pixel colour(double px, double py, double t) const {
        double u, v;
        const int i = visible(px, py, t, u, v);
        if (i < 0) return {};
        const Object& o = objects[size_t(i)];
        return texture(o.tex, u, v, o.seed);
    }
    // True displacement from time 0 to time t of what pixel (px, py) shows at time 0.
    Vec2 flow(double px, double py, double t) const {
        double u, v;
        const int i = visible(px, py, 0, u, v);
        if (i < 0) return {};
        double qx, qy;
        objects[size_t(i)].place(u, v, t, qx, qy);
        return {float(qx - px), float(qy - py)};
    }
    Scene scaled(double k) const {
        Scene s = *this;
        s.width = int(width * k);
        s.height = int(height * k);
        for (auto& o : s.objects) {
            o.cx *= k;
            o.cy *= k;
            o.vx *= k;
            o.vy *= k;
            o.hw *= k;
            o.hh *= k;
            o.corner *= k;
        }
        return s;
    }
};

Object background(double vx, double vy, uint32_t seed, double spin = 0, double cx = 320, double cy = 180) {
    Object o;
    o.background = true;
    o.vx = vx;
    o.vy = vy;
    o.spin = spin;
    o.cx = cx;
    o.cy = cy;
    o.seed = seed;
    return o;
}

Object card(double cx, double cy, double hw, double hh, double vx, double vy, double spin, uint32_t seed,
            double corner = 12) {
    Object o;
    o.cx = cx;
    o.cy = cy;
    o.hw = hw;
    o.hh = hh;
    o.corner = std::min(corner, std::min(hw, hh));
    o.vx = vx;
    o.vy = vy;
    o.spin = spin;
    o.tex = 1;
    o.seed = seed;
    return o;
}

std::vector<Scene> scenes() {
    std::vector<Scene> s;
    {
        Scene c;
        c.name = "object";
        c.what = "card moving right and turning over a background panning left";
        c.objects = {background(-5, 1.5, 11), card(300, 190, 80, 55, 22, -3, 4 * kPi / 180, 21)};
        s.push_back(c);
    }
    {
        Scene c;
        c.name = "pan_follow";
        c.what = "camera follows a still card: background pans fast behind it";
        c.objects = {background(34, 0, 12), card(320, 180, 70, 95, 0, 0, 0, 22)};
        s.push_back(c);
    }
    {
        Scene c;
        c.name = "thin";
        c.what = "thin pole crossing fast over a still background";
        c.objects = {background(0, 0, 13), card(300, 180, 4, 150, 40, 0, 0, 23, 2)};
        s.push_back(c);
    }
    {
        Scene c;
        c.name = "edge";
        c.what = "card leaving the frame fast on the right";
        c.objects = {background(0, 0, 14), card(600, 140, 90, 60, 48, 8, 0, 24)};
        s.push_back(c);
    }
    {
        Scene c;
        c.name = "crossing";
        c.what = "two cards crossing in opposite directions over a slow pan";
        c.objects = {background(3, 0, 15), card(250, 170, 70, 50, 18, 2, 0, 25),
                     card(390, 200, 60, 70, -20, -2, -3 * kPi / 180, 26)};
        s.push_back(c);
    }
    {
        Scene c = s[0];
        c.name = "object_grain";
        c.what = "the first scene with film grain (σ = 3% per channel, new each frame)";
        c.grain = 0.03f;
        s.push_back(c);
    }
    {
        Scene c;
        c.name = "static";
        c.what = "nothing moves";
        c.objects = {background(0, 0, 16), card(320, 180, 90, 60, 0, 0, 0, 27)};
        s.push_back(c);
    }
    return s;
}

// --- Rendering the scenes -------------------------------------------------------------

Image render_instant(const Scene& sc, double t, int ss) {
    Image img(sc.width, sc.height);
    smear::parallel_for(sc.height, [&](int y) {
        for (int x = 0; x < sc.width; ++x) {
            float r = 0, g = 0, b = 0, a = 0;
            for (int j = 0; j < ss; ++j)
                for (int i = 0; i < ss; ++i) {
                    const Pixel p = sc.colour(x + (i + 0.5) / ss - 0.5, y + (j + 0.5) / ss - 0.5, t);
                    r += p.r;
                    g += p.g;
                    b += p.b;
                    a += p.a;
                }
            const float k = 1.0f / float(ss * ss);
            img.at(x, y) = {r * k, g * k, b * k, a * k};
        }
    });
    return img;
}

// Gaussian-ish grain (sum of uniforms), a new pattern per `frame`.
void add_grain(Image& img, float sigma, uint32_t frame) {
    if (sigma <= 0) return;
    for (int y = 0; y < img.height; ++y)
        for (int x = 0; x < img.width; ++x) {
            Pixel& p = img.at(x, y);
            float* ch[3] = {&p.r, &p.g, &p.b};
            for (int c = 0; c < 3; ++c) {
                double u = 0;
                for (int k = 0; k < 4; ++k) u += lattice(x, y, frame * 977u + uint32_t(c * 4 + k) * 131u) - 0.5;
                *ch[c] += float(u * std::sqrt(3.0)) * sigma;  // 4 uniforms: variance 1/3
            }
        }
}

Image render_truth(const Scene& sc, double shutter, int subframes = 64) {
    Image img(sc.width, sc.height);
    const int grid = 8;  // sub-pixel offsets, one per subframe (shuffled)
    smear::parallel_for(sc.height, [&](int y) {
        for (int x = 0; x < sc.width; ++x) {
            float r = 0, g = 0, b = 0, a = 0;
            for (int k = 0; k < subframes; ++k) {
                const double t = ((k + 0.5) / subframes - 0.5) * shutter;
                const int cell = (k * 37 + (x * 7 + y * 13) % 64) % (grid * grid);
                const double ox = (cell % grid + 0.5) / grid - 0.5, oy = (cell / grid + 0.5) / grid - 0.5;
                const Pixel p = sc.colour(x + ox, y + oy, t);
                r += p.r;
                g += p.g;
                b += p.b;
                a += p.a;
            }
            const float kk = 1.0f / float(subframes);
            img.at(x, y) = {r * kk, g * kk, b * kk, a * kk};
        }
    });
    return img;
}

FlowField true_flow(const Scene& sc, double t) {
    FlowField f(sc.width, sc.height);
    smear::parallel_for(sc.height, [&](int y) {
        for (int x = 0; x < sc.width; ++x) f.at(x, y) = sc.flow(x, y, t);
    });
    return f;
}

// Occlusion band: pixels whose visible layer changes between frames −1 and +1,
// plus pixels near a layer boundary at frame 0 (within the longest blur reach),
// where background and foreground blurs interact.
std::vector<uint8_t> occlusion_band(const Scene& sc, const FlowField& tn, double shutter) {
    const int w = sc.width, h = sc.height;
    std::vector<int> layer(size_t(w) * size_t(h));
    float vmax = 0;
    for (const Vec2& v : tn.px) vmax = std::max(vmax, std::sqrt(v.x * v.x + v.y * v.y));
    const int r = int(std::ceil(0.5 * shutter * vmax)) + 3;
    std::vector<uint8_t> band(layer.size());
    smear::parallel_for(h, [&](int y) {
        for (int x = 0; x < w; ++x) {
            double u, v;
            const int base = sc.visible(x, y, 0, u, v);
            layer[size_t(y) * size_t(w) + size_t(x)] = base;
            bool change = false;
            for (int k = 0; k <= 20 && !change; ++k) change = sc.visible(x, y, -1.0 + k * 0.1, u, v) != base;
            band[size_t(y) * size_t(w) + size_t(x)] = change ? 1 : 0;
        }
    });
    smear::parallel_for(h, [&](int y) {
        for (int x = 0; x < w; ++x) {
            const int base = layer[size_t(y) * size_t(w) + size_t(x)];
            bool edge = false;
            for (int dy = -r; dy <= r && !edge; dy += 2)
                for (int dx = -r; dx <= r && !edge; dx += 2) {
                    const int xx = std::clamp(x + dx, 0, w - 1), yy = std::clamp(y + dy, 0, h - 1);
                    edge = layer[size_t(yy) * size_t(w) + size_t(xx)] != base;
                }
            if (edge) band[size_t(y) * size_t(w) + size_t(x)] = 1;
        }
    });
    return band;
}

// Layer index at frame 0 (0 = background).
std::vector<int> layers(const Scene& sc) {
    std::vector<int> l(size_t(sc.width) * size_t(sc.height));
    for (int y = 0; y < sc.height; ++y)
        for (int x = 0; x < sc.width; ++x) {
            double u, v;
            l[size_t(y) * size_t(sc.width) + size_t(x)] = sc.visible(x, y, 0, u, v);
        }
    return l;
}

// --- Baseline: blur along the flow, no occlusion handling ------------------------------

Image naive_blur(const Image& cur, const FlowField& to_prev, const FlowField& to_next, double shutter,
                 int max_samples) {
    Image out(cur.width, cur.height);
    smear::parallel_for(cur.height, [&](int y) {
        for (int x = 0; x < cur.width; ++x) {
            const Vec2 un = to_next.at(x, y), up = to_prev.at(x, y);
            const float hx = 0.5f * (un.x - up.x) * float(shutter) * 0.5f;
            const float hy = 0.5f * (un.y - up.y) * float(shutter) * 0.5f;
            const float len = std::sqrt(hx * hx + hy * hy);
            if (len < 0.5f) {
                out.at(x, y) = cur.at(x, y);
                continue;
            }
            const int n = std::min(max_samples, std::max(2, 2 * int(std::ceil(len))));
            float r = 0, g = 0, b = 0, a = 0;
            for (int i = 0; i < n; ++i) {
                const float s = -1.0f + (2.0f * float(i) + 1.0f) / float(n);
                const Pixel p = smear::sample_image(cur, float(x) + s * hx, float(y) + s * hy);
                r += p.r;
                g += p.g;
                b += p.b;
                a += p.a;
            }
            const float k = 1.0f / float(n);
            out.at(x, y) = {r * k, g * k, b * k, a * k};
        }
    });
    return out;
}

// --- Metrics ---------------------------------------------------------------------------

struct Err {
    double mae = 0, psnr = 0, band_mae = 0;
};

Err compare(const Image& a, const Image& b, const std::vector<uint8_t>& band) {
    double se = 0, ae = 0, bae = 0;
    size_t nb = 0;
    for (size_t i = 0; i < a.px.size(); ++i) {
        const Pixel &p = a.px[i], &q = b.px[i];
        const double d[3] = {double(p.r - q.r), double(p.g - q.g), double(p.b - q.b)};
        double s = 0, m = 0;
        for (double v : d) {
            s += v * v;
            m += std::fabs(v);
        }
        se += s / 3;
        ae += m / 3;
        if (band[i]) {
            bae += m / 3;
            ++nb;
        }
    }
    const double n = double(a.px.size());
    Err e;
    e.mae = ae / n * 255.0;
    e.psnr = se > 0 ? 10.0 * std::log10(n / se) : 99.0;
    e.band_mae = nb ? bae / double(nb) * 255.0 : 0.0;
    return e;
}

struct Epe {
    double all = 0, band = 0;
};

Epe flow_error(const FlowField& f, const FlowField& truth, const std::vector<uint8_t>& band) {
    double s = 0, sb = 0;
    size_t nb = 0;
    for (size_t i = 0; i < f.px.size(); ++i) {
        const double dx = f.px[i].x - truth.px[i].x, dy = f.px[i].y - truth.px[i].y;
        const double e = std::sqrt(dx * dx + dy * dy);
        s += e;
        if (band[i]) {
            sb += e;
            ++nb;
        }
    }
    return {s / double(f.px.size()), nb ? sb / double(nb) : 0.0};
}

bool write_ppm(const std::string& path, const Image& img) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fprintf(f, "P6\n%d %d\n255\n", img.width, img.height);
    for (const auto& p : img.px) {
        auto c = [](float v) { return (unsigned char)(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f); };
        const unsigned char rgb[3] = {c(p.r), c(p.g), c(p.b)};
        std::fwrite(rgb, 1, 3, f);
    }
    std::fclose(f);
    return true;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    std::string out_dir = "smear_bench_out", only;
    bool images = true, timing = true, debug = false, use_true = false;
    int quality = 2;
    double sensitivity = 50;
    std::vector<std::pair<int, int>> probes;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--no-images")) images = false;
        else if (!std::strcmp(argv[i], "--no-timing")) timing = false;
        else if (!std::strcmp(argv[i], "--debug")) debug = true;
        else if (!std::strcmp(argv[i], "--true-flow")) use_true = true;
        else if (!std::strcmp(argv[i], "--quality") && i + 1 < argc) quality = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--sensitivity") && i + 1 < argc) sensitivity = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--probe") && i + 2 < argc) {
            probes.push_back({std::atoi(argv[i + 1]), std::atoi(argv[i + 2])});
            i += 2;
        }
        else if (!std::strcmp(argv[i], "--scene") && i + 1 < argc) only = argv[++i];
        else out_dir = argv[i];
    }

    if (images || debug) {
        std::error_code ec;
        std::filesystem::create_directories(out_dir, ec);
    }
    const double shutter = 0.5;
    smear::FlowSettings fs;
    fs.quality = smear::Quality(quality);
    fs.sensitivity = sensitivity;
    smear::BlurSettings bs;
    bs.shutter = shutter;

    const char* qnames[4] = {"", "Draft", "Normal", "Best"};
    std::printf("Shutter %.0f°, flow: sensitivity %.0f, %s quality, %d samples. Errors in 8-bit levels.\n\n",
                shutter * 360, fs.sensitivity, qnames[std::clamp(quality, 1, 3)], bs.max_samples);
    std::printf("| scene | flow EPE all / band (px) | PSNR naive / smear / smear·true-flow (dB) | "
                "band error naive / smear / smear·true-flow |\n");
    std::printf("| --- | --- | --- | --- |\n");
    double sum_naive = 0, sum_smear = 0;
    int scenes_run = 0;
    for (const Scene& sc : scenes()) {
        if (!only.empty() && sc.name != only) continue;
        Image prev = render_instant(sc, -1, 4), cur = render_instant(sc, 0, 4), next = render_instant(sc, 1, 4);
        Image truth = render_truth(sc, shutter);
        add_grain(prev, sc.grain, 1);
        add_grain(cur, sc.grain, 2);
        add_grain(truth, sc.grain, 2);  // the grain of frame 0, unblurred, as a camera would record it
        add_grain(next, sc.grain, 3);
        const FlowField tp = true_flow(sc, -1), tn = true_flow(sc, 1);
        const std::vector<uint8_t> band = occlusion_band(sc, tn, shutter);

        FlowField fp, fn;
        smear::compute_flow(cur, prev, fp, fs);
        smear::compute_flow(cur, next, fn, fs);
        if (use_true) {
            fp = tp;
            fn = tn;
        }

        const Image naive = naive_blur(cur, fp, fn, shutter, bs.max_samples);
        Image ours, ours_true, vectors;
        smear::render_motion_blur(cur, &prev, &next, &fp, &fn, ours, bs);
        smear::render_motion_blur(cur, &prev, &next, &tp, &tn, ours_true, bs);

        const Epe en = flow_error(fn, tn, band), ep = flow_error(fp, tp, band);
        const Err a = compare(naive, truth, band), b = compare(ours, truth, band), c = compare(ours_true, truth, band);
        sum_naive += a.band_mae;
        sum_smear += b.band_mae;
        ++scenes_run;
        std::printf("| %s | %.2f / %.2f | %.1f / %.1f / %.1f | %.2f / %.2f / %.2f |\n", sc.name.c_str(),
                    0.5 * (en.all + ep.all), 0.5 * (en.band + ep.band), a.psnr, b.psnr, c.psnr, a.band_mae,
                    b.band_mae, c.band_mae);
        std::fflush(stdout);

        if (debug) {
            // Motion vector error (vs true centred velocity) and layer classification in the band.
            smear::BlurBuffers bb;
            const smear::Inputs in = smear::select_inputs(cur, &prev, &next, &fp, &fn, bs.track);
            smear::prepare_blur(cur, in, bs, smear::BlurTuning{}, bb);
            const std::vector<int> lay = layers(sc);
            const float half = float(shutter) * 0.5f;
            double mv = 0, mvb = 0, wrong = 0;
            size_t nb = 0;
            Image occ(sc.width, sc.height), front(sc.width, sc.height), err(sc.width, sc.height);
            for (int y = 0; y < sc.height; ++y)
                for (int x = 0; x < sc.width; ++x) {
                    const size_t i = size_t(y) * size_t(sc.width) + size_t(x);
                    const smear::Vec4 m = bb.motion.at(x, y);
                    const float tx = 0.5f * (tn.px[i].x - tp.px[i].x), ty = 0.5f * (tn.px[i].y - tp.px[i].y);
                    const float ex = m.x / half - tx, ey = m.y / half - ty;
                    const double e = std::sqrt(ex * ex + ey * ey);
                    mv += e;
                    const smear::Background bk = smear::background_at(bb.neighbour_occ, bb.gather.tile, smear::BlurTuning{}, x, y);
                    const float z = smear::front_score(m.x, m.y, bk.x, bk.y, bb.gather.to_velocity, bk.gate, smear::BlurTuning{});
                    if (band[i]) {
                        mvb += e;
                        wrong += std::fabs(double(z) - (lay[i] > 0 ? 1.0 : 0.0));
                        ++nb;
                    }
                    occ.px[i] = {m.z, m.z, m.z, 1};
                    front.px[i] = {z, band[i] ? 0.5f : 0.0f, lay[i] > 0 ? 1.0f : 0.0f, 1};
                    const float ee = float(std::min(1.0, e / 8.0));
                    err.px[i] = {ee, ee, ee, 1};
                }
            std::printf("|   (debug) | motion EPE all %.2f band %.2f | layer misclassified in band %.1f%% |  |\n",
                        mv / double(occ.px.size()), nb ? mvb / double(nb) : 0.0, nb ? 100.0 * wrong / double(nb) : 0.0);
            {
                double so[2] = {0, 0}, sh[2] = {0, 0};
                for (int y = 0; y < sc.height; ++y)
                    for (int x = 0; x < sc.width; ++x) {
                        const smear::Vec4 m = bb.motion.at(x, y);
                        const int k = lay[size_t(y) * size_t(sc.width) + size_t(x)] > 0 ? 1 : 0;
                        so[k] += m.z;
                        sh[k] += m.z * std::sqrt(m.x * m.x + m.y * m.y);
                    }
                std::printf("occ mass bg %.0f (mean |h| %.2f), fg %.0f (mean |h| %.2f)\n", so[0], sh[0] / std::max(so[0], 1e-9),
                            so[1], sh[1] / std::max(so[1], 1e-9));
            }
            for (const auto& pr : probes) {
                const int x = pr.first, y = pr.second;
                const size_t i = size_t(y) * size_t(sc.width) + size_t(x);
                const smear::Vec4 m = bb.motion.at(x, y);
                const smear::Background bk = smear::background_at(bb.neighbour_occ, bb.gather.tile, smear::BlurTuning{}, x, y);
                std::printf("probe %d,%d: fn %.2f,%.2f tn %.2f,%.2f | fp %.2f,%.2f tp %.2f,%.2f | en %.3f ep %.3f | h %.2f,%.2f occ %.2f | bg %.2f,%.2f gate %.2f\n",
                    x, y, fn.px[i].x, fn.px[i].y, tn.px[i].x, tn.px[i].y, fp.px[i].x, fp.px[i].y, tp.px[i].x, tp.px[i].y,
                    smear::match_error(cur.px[i], next, float(x) + fn.px[i].x, float(y) + fn.px[i].y),
                    smear::match_error(cur.px[i], prev, float(x) + fp.px[i].x, float(y) + fp.px[i].y), m.x, m.y, m.z, bk.x, bk.y, bk.gate);
            }
            const std::string p = out_dir + "/" + sc.name;
            write_ppm(p + "_occ.ppm", occ);
            write_ppm(p + "_layers.ppm", front);  // red: judged in front, green: band, blue: truly in front
            write_ppm(p + "_mverr.ppm", err);
        }
        if (images) {
            smear::BlurSettings vs = bs;
            vs.show_vectors = true;
            smear::render_motion_blur(cur, &prev, &next, &fp, &fn, vectors, vs);
            const std::string p = out_dir + "/" + sc.name;
            write_ppm(p + "_frame.ppm", cur);
            write_ppm(p + "_truth.ppm", truth);
            write_ppm(p + "_naive.ppm", naive);
            write_ppm(p + "_smear.ppm", ours);
            write_ppm(p + "_vectors.ppm", vectors);
        }
    }
    if (scenes_run)
        std::printf("\nBand error averaged over the scenes: naive %.2f, smear %.2f (%.0f%% lower)\n",
                    sum_naive / scenes_run, sum_smear / scenes_run, 100.0 * (1.0 - sum_smear / std::max(sum_naive, 1e-9)));

    if (timing) {
        const Scene big = scenes()[0].scaled(3.0);  // 1920×1080
        const Image prev = render_instant(big, -1, 1), cur = render_instant(big, 0, 1), next = render_instant(big, 1, 1);
        std::printf("\nCPU time per frame at %dx%d (%u threads):\n\n", big.width, big.height,
                    std::thread::hardware_concurrency());
        std::printf("| quality | 2 flows (s) | blur (s) | total (s) |\n| --- | --- | --- | --- |\n");
        const smear::Quality qs[3] = {smear::Quality::Draft, smear::Quality::Normal, smear::Quality::Best};
        const char* names[3] = {"Draft", "Normal", "Best"};
        for (int q = 0; q < 3; ++q) {
            smear::FlowSettings f = fs;
            f.quality = qs[q];
            FlowField fp, fn;
            Image out;
            const auto t0 = std::chrono::steady_clock::now();
            smear::compute_flow(cur, prev, fp, f);
            smear::compute_flow(cur, next, fn, f);
            const double tf = seconds_since(t0);
            const auto t1 = std::chrono::steady_clock::now();
            smear::render_motion_blur(cur, &prev, &next, &fp, &fn, out, bs);
            const double tb = seconds_since(t1);
            std::printf("| %s | %.3f | %.3f | %.3f |\n", names[q], tf, tb, tf + tb);
            std::fflush(stdout);
        }
    }
    return 0;
}
