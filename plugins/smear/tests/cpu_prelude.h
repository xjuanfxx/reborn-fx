// cpu_prelude.h — runs Smear's GPU kernel source (smear_kernels.inc) on the CPU.
//
// Defines the kernel macros so each kernel becomes a C++ function called over
// its w × h grid (rows in parallel), registered by name, and provides a CPU
// Backend/Scratch for SmearGpuRender.h. The GPU passes and their orchestration
// can then be checked against the CPU renderer without a GPU.
//
// Images in the GPU layout are float4 BGRA; to_gpu()/from_gpu() convert.
#pragma once

#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "SmearGpuRender.h"
#include "smear_gpu_types.h"

namespace smear_cpu {

struct float2 {
    float x, y;
};
struct float4 {
    float x, y, z, w;
};
inline float4 operator+(float4 a, float4 b) { return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; }
inline float4 operator-(float4 a, float4 b) { return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w}; }
inline float4 operator*(float4 a, float k) { return {a.x * k, a.y * k, a.z * k, a.w * k}; }

using std::atan2;
using std::ceil;
using std::exp;
using std::fabs;
using std::floor;
using std::fmax;
using std::fmin;
using std::sqrt;
inline int min(int a, int b) { return a < b ? a : b; }
inline int max(int a, int b) { return a > b ? a : b; }

// Kernel launcher: grid size, buffers in order, parameter block.
using Launch = void (*)(int w, int h, void* const* buffers, const void* params);

inline std::map<std::string, Launch>& registry() {
    static std::map<std::string, Launch> r;
    return r;
}
inline bool add(const char* name, Launch fn) {
    registry()[name] = fn;
    return true;
}

}  // namespace smear_cpu

#define DEVICE static inline
#define PRIV
#define GBUF_CONST const float4*
#define GBUF float4*
#define GBUF2_CONST const float2*
#define GBUF2 float2*
#define F4(a, b, c, d) float4{(a), (b), (c), (d)}
#define F2(a, b) float2{(a), (b)}

// Each kernel: a body function per thread, a launcher over the grid (rows in
// parallel) and a registration by name.
#define KERNEL_B2(name, P, T0, a0, T1, a1)                                         \
    static void name##_body(const int x, const int y, T0 a0, T1 a1, const P& p);   \
    static void name##_launch(int w, int h, void* const* b, const void* pp) {      \
        const P& p = *static_cast<const P*>(pp);                                   \
        smear::parallel_for(h, [&](int y) {                                        \
            for (int x = 0; x < w; ++x) name##_body(x, y, (T0)b[0], (T1)b[1], p);  \
        });                                                                        \
    }                                                                              \
    static const bool name##_registered = add(#name, &name##_launch);              \
    static void name##_body(const int x, const int y, T0 a0, T1 a1, const P& p) {

#define KERNEL_B3(name, P, T0, a0, T1, a1, T2, a2)                                           \
    static void name##_body(const int x, const int y, T0 a0, T1 a1, T2 a2, const P& p);      \
    static void name##_launch(int w, int h, void* const* b, const void* pp) {                \
        const P& p = *static_cast<const P*>(pp);                                             \
        smear::parallel_for(h, [&](int y) {                                                  \
            for (int x = 0; x < w; ++x) name##_body(x, y, (T0)b[0], (T1)b[1], (T2)b[2], p);  \
        });                                                                                  \
    }                                                                                        \
    static const bool name##_registered = add(#name, &name##_launch);                        \
    static void name##_body(const int x, const int y, T0 a0, T1 a1, T2 a2, const P& p) {

#define KERNEL_B4(name, P, T0, a0, T1, a1, T2, a2, T3, a3)                                             \
    static void name##_body(const int x, const int y, T0 a0, T1 a1, T2 a2, T3 a3, const P& p);         \
    static void name##_launch(int w, int h, void* const* b, const void* pp) {                          \
        const P& p = *static_cast<const P*>(pp);                                                       \
        smear::parallel_for(h, [&](int y) {                                                            \
            for (int x = 0; x < w; ++x) name##_body(x, y, (T0)b[0], (T1)b[1], (T2)b[2], (T3)b[3], p);  \
        });                                                                                            \
    }                                                                                                  \
    static const bool name##_registered = add(#name, &name##_launch);                                  \
    static void name##_body(const int x, const int y, T0 a0, T1 a1, T2 a2, T3 a3, const P& p) {

#define KERNEL_B5(name, P, T0, a0, T1, a1, T2, a2, T3, a3, T4, a4)                                               \
    static void name##_body(const int x, const int y, T0 a0, T1 a1, T2 a2, T3 a3, T4 a4, const P& p);            \
    static void name##_launch(int w, int h, void* const* b, const void* pp) {                                    \
        const P& p = *static_cast<const P*>(pp);                                                                 \
        smear::parallel_for(h, [&](int y) {                                                                      \
            for (int x = 0; x < w; ++x) name##_body(x, y, (T0)b[0], (T1)b[1], (T2)b[2], (T3)b[3], (T4)b[4], p);  \
        });                                                                                                      \
    }                                                                                                            \
    static const bool name##_registered = add(#name, &name##_launch);                                            \
    static void name##_body(const int x, const int y, T0 a0, T1 a1, T2 a2, T3 a3, T4 a4, const P& p) {

#define KERNEL_B6(name, P, T0, a0, T1, a1, T2, a2, T3, a3, T4, a4, T5, a5)                                                 \
    static void name##_body(const int x, const int y, T0 a0, T1 a1, T2 a2, T3 a3, T4 a4, T5 a5, const P& p);               \
    static void name##_launch(int w, int h, void* const* b, const void* pp) {                                              \
        const P& p = *static_cast<const P*>(pp);                                                                           \
        smear::parallel_for(h, [&](int y) {                                                                                \
            for (int x = 0; x < w; ++x) name##_body(x, y, (T0)b[0], (T1)b[1], (T2)b[2], (T3)b[3], (T4)b[4], (T5)b[5], p);  \
        });                                                                                                                \
    }                                                                                                                      \
    static const bool name##_registered = add(#name, &name##_launch);                                                      \
    static void name##_body(const int x, const int y, T0 a0, T1 a1, T2 a2, T3 a3, T4 a4, T5 a5, const P& p) {

#define KERNEL_END }

namespace smear_cpu {
#include "smear_kernels.inc"
}  // namespace smear_cpu

#undef DEVICE
#undef PRIV
#undef GBUF_CONST
#undef GBUF
#undef GBUF2_CONST
#undef GBUF2
#undef F4
#undef F2
#undef KERNEL_B2
#undef KERNEL_B3
#undef KERNEL_B4
#undef KERNEL_B5
#undef KERNEL_B6
#undef KERNEL_END

namespace smear_cpu {

// Host memory standing in for GPU buffers. Rows are padded so pitches differ
// from widths, which catches pitch mix-ups.
class CpuScratch final : public smear::gpu::Scratch {
public:
    bool alloc(int w, int h, int channels, smear::gpu::Buffer& out) override {
        if (w <= 0 || h <= 0 || (channels != 2 && channels != 4)) return false;
        const int pitch = w + 3;
        blocks_.emplace_back(new float[size_t(pitch) * size_t(h) * size_t(channels)]());
        out = {blocks_.back().get(), w, h, pitch};
        ++count_;
        return true;
    }
    int count() const { return count_; }

private:
    std::vector<std::unique_ptr<float[]>> blocks_;
    int count_ = 0;
};

class CpuBackend final : public smear::gpu::Backend {
public:
    bool run(const char* name, int w, int h, std::initializer_list<void*> buffers, const void* params,
             size_t params_size) override {
        (void)params_size;
        const auto it = registry().find(name);
        if (it == registry().end()) {
            missing = name;
            return false;
        }
        std::vector<void*> b(buffers);
        for (void* p : b)
            if (!p) return false;
        it->second(w, h, b.data(), params);
        ++passes;
        return true;
    }
    int passes = 0;
    std::string missing;
};

// A float4 BGRA buffer holding a CPU image, with padded rows.
struct GpuImage {
    std::vector<float4> data;
    smear::gpu::Buffer buf{nullptr, 0, 0, 0};

    explicit GpuImage(const smear::Image& img, int pad = 5) {
        const int pitch = img.width + pad;
        data.assign(size_t(pitch) * size_t(img.height), float4{0, 0, 0, 0});
        for (int y = 0; y < img.height; ++y)
            for (int x = 0; x < img.width; ++x) {
                const smear::Pixel& p = img.at(x, y);
                data[size_t(y) * size_t(pitch) + size_t(x)] = {p.b, p.g, p.r, p.a};
            }
        buf = {data.data(), img.width, img.height, pitch};
    }
    GpuImage(int w, int h, int pad = 7) {
        const int pitch = w + pad;
        data.assign(size_t(pitch) * size_t(h), float4{0, 0, 0, 0});
        buf = {data.data(), w, h, pitch};
    }
    smear::Image image() const {
        smear::Image img(buf.width, buf.height);
        for (int y = 0; y < buf.height; ++y)
            for (int x = 0; x < buf.width; ++x) {
                const float4& v = data[size_t(y) * size_t(buf.pitch) + size_t(x)];
                img.at(x, y) = {v.z, v.y, v.x, v.w};
            }
        return img;
    }
};

// A float2 buffer (flow field) with padded rows.
struct GpuFlow {
    std::vector<float2> data;
    smear::gpu::Buffer buf{nullptr, 0, 0, 0};

    GpuFlow(int w, int h, int pad = 3) {
        const int pitch = w + pad;
        data.assign(size_t(pitch) * size_t(h), float2{0, 0});
        buf = {data.data(), w, h, pitch};
    }
    smear::FlowField field() const {
        smear::FlowField f(buf.width, buf.height);
        for (int y = 0; y < buf.height; ++y)
            for (int x = 0; x < buf.width; ++x) {
                const float2& v = data[size_t(y) * size_t(buf.pitch) + size_t(x)];
                f.at(x, y) = {v.x, v.y};
            }
        return f;
    }
};

}  // namespace smear_cpu
