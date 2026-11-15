// render_frames — renders frames with the CPU renderer and writes them as PPM.
//
// Usage: analog_render_frames <out_dir> [input.ppm] [frames]
// Without an input image a test pattern (color bars and a gradient) is used.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "analog_cpu.h"

using analog::float4;

namespace {

bool read_ppm(const char* path, int& w, int& h, std::vector<float4>& img) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    int maxv = 0;
    if (std::fscanf(f, "P6 %d %d %d", &w, &h, &maxv) != 3 || maxv != 255) {
        std::fclose(f);
        return false;
    }
    std::fgetc(f);
    std::vector<unsigned char> rgb(size_t(w) * h * 3);
    const bool ok = std::fread(rgb.data(), 1, rgb.size(), f) == rgb.size();
    std::fclose(f);
    img.resize(size_t(w) * h);
    for (size_t i = 0; i < img.size(); ++i)
        img[i] = {rgb[3 * i + 2] / 255.0f, rgb[3 * i + 1] / 255.0f, rgb[3 * i] / 255.0f, 1.0f};
    return ok;
}

void write_ppm(const std::string& path, int w, int h, const std::vector<float4>& img) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    std::fprintf(f, "P6 %d %d 255\n", w, h);
    std::vector<unsigned char> rgb(size_t(w) * h * 3);
    auto q = [](float v) { return (unsigned char)(std::min(1.0f, std::max(0.0f, v)) * 255.0f + 0.5f); };
    for (size_t i = 0; i < img.size(); ++i) {
        rgb[3 * i] = q(img[i].z);
        rgb[3 * i + 1] = q(img[i].y);
        rgb[3 * i + 2] = q(img[i].x);
    }
    std::fwrite(rgb.data(), 1, rgb.size(), f);
    std::fclose(f);
}

std::vector<float4> pattern(int w, int h) {
    static const float bars[7][3] = {{.75f, .75f, .75f}, {.75f, .75f, 0}, {0, .75f, .75f}, {0, .75f, 0},
                                     {.75f, 0, .75f},    {.75f, 0, 0},    {0, 0, .75f}};
    std::vector<float4> img(size_t(w) * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float r, g, b;
            if (y < h * 2 / 3) {
                const float* c = bars[x * 7 / w];
                r = c[0], g = c[1], b = c[2];
            } else {
                r = g = b = float(x) / w;
            }
            img[size_t(y) * w + x] = {b, g, r, 1.0f};
        }
    return img;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("Usage: analog_render_frames <out_dir> [input.ppm] [frames]\n");
        return 2;
    }
    int w = 640, h = 360;
    std::vector<float4> in;
    if (argc >= 3 && std::string(argv[2]) != "-") {
        if (!read_ppm(argv[2], w, h, in)) {
            std::printf("Cannot read %s (binary P6, 8-bit)\n", argv[2]);
            return 1;
        }
    } else {
        in = pattern(w, h);
    }
    const int frames = argc >= 4 ? std::atoi(argv[3]) : 3;
    analog::Frame f;
    f.canvas_w = f.in_w = f.out_w = w;
    f.rows = f.in_h = f.out_h = h;
    f.time_scale = 30;
    std::vector<float4> out(in.size());
    for (int i = 0; i < frames; ++i) {
        f.time = i;
        analog::render_cpu(analog::make_params(analog::Params{}, f), in.data(), out.data());
        const std::string path = std::string(argv[1]) + "/frame_" + std::to_string(i) + ".ppm";
        write_ppm(path, w, h, out);
        std::printf("%s\n", path.c_str());
    }
    return 0;
}
