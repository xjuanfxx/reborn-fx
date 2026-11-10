// render_frames — renders Tremor on a test pattern (or a binary PPM) with the
// default parameters and writes the frames as PPM images, without After Effects.
//
// Usage: render_frames <out_dir> [frames] [input.ppm]
#include <cstdio>
#include <cstdlib>
#include <string>

#include "tremor_render.h"

namespace {

tremor::Image pattern(int w, int h) {
    tremor::Image img(w, h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const bool check = ((x / 40) + (y / 40)) % 2 == 0;
            const float v = check ? 0.85f : 0.25f;
            img.at(x, y) = {v * float(x) / w + 0.1f, v, v * float(y) / h + 0.1f, 1.0f};
        }
    return img;
}

bool read_ppm(const char* path, tremor::Image& img) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    int w = 0, h = 0, maxv = 0;
    const bool ok = std::fscanf(f, "P6 %d %d %d", &w, &h, &maxv) == 3 && maxv == 255 && std::fgetc(f) != EOF;
    if (ok) {
        img = tremor::Image(w, h);
        for (auto& p : img.px) {
            unsigned char c[3] = {};
            if (std::fread(c, 1, 3, f) != 3) break;
            p = {c[0] / 255.0f, c[1] / 255.0f, c[2] / 255.0f, 1.0f};
        }
    }
    std::fclose(f);
    return ok;
}

bool write_ppm(const std::string& path, const tremor::Image& img) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fprintf(f, "P6\n%d %d\n255\n", img.width, img.height);
    for (const auto& p : img.px) {
        auto c = [&](float v) { return (unsigned char)(std::clamp(v * p.a, 0.0f, 1.0f) * 255.0f + 0.5f); };
        const unsigned char rgb[3] = {c(p.r), c(p.g), c(p.b)};
        std::fwrite(rgb, 1, 3, f);
    }
    std::fclose(f);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("Usage: render_frames <out_dir> [frames] [input.ppm]\n");
        return 2;
    }
    const std::string dir = argv[1];
    const int frames = argc > 2 ? std::atoi(argv[2]) : 30;
    tremor::Image src = pattern(640, 360);
    if (argc > 3 && !read_ppm(argv[3], src)) {
        std::printf("Cannot read %s (binary PPM, 8 bit)\n", argv[3]);
        return 1;
    }
    const tremor::Params p;
    for (int f = 0; f < frames; ++f) {
        const tremor::FrameState st = tremor::evaluate(p, f * 1001, 30000, src.width, src.height);
        tremor::Image out(src.width, src.height);
        tremor::render_frame(src, out, 0, 0, st, tremor::make_look(p, 8));
        char name[64];
        std::snprintf(name, sizeof(name), "/frame_%03d.ppm", f);
        if (!write_ppm(dir + name, out)) {
            std::printf("Cannot write %s%s\n", dir.c_str(), name);
            return 1;
        }
    }
    std::printf("Wrote %d frames to %s\n", frames, dir.c_str());
    return 0;
}
