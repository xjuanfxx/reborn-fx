// gpu_check — runs Reception's GPU kernels on this machine and compares them with
// the CPU renderer, without After Effects. Also times both paths at 1080p.
//
// Creates its own CUDA (driver API) and OpenCL contexts.
//
// Usage: reception_gpu_check            (needs an NVIDIA and/or OpenCL GPU)
#include <windows.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "ReceptionGpuRender.h"
#include "reception_cpu.h"

namespace {

using reception::float4;

template <class Fn>
bool load_fn(HMODULE dll, Fn& fn, const char* name) {
    fn = reinterpret_cast<Fn>(GetProcAddress(dll, name));
    return fn != nullptr;
}

// A GPU API seen by the test: memory plus a Reception backend.
class Device {
public:
    virtual ~Device() = default;
    virtual const char* name() const = 0;
    virtual reception::gpu::Backend& backend() = 0;
    virtual void* alloc(size_t bytes) = 0;
    virtual void release(void* mem) = 0;
    virtual bool upload(void* mem, const void* data, size_t bytes) = 0;
    virtual bool download(void* mem, void* data, size_t bytes) = 0;
    virtual bool finish() = 0;
};

class DeviceScratch final : public reception::gpu::Scratch {
public:
    explicit DeviceScratch(Device& d) : device_(d) {}
    ~DeviceScratch() override {
        for (void* m : buffers_) device_.release(m);
    }
    bool alloc(int w, int h, reception::gpu::Buffer& out) override {
        void* m = device_.alloc(size_t(w) * h * 16);
        if (!m) return false;
        buffers_.push_back(m);
        out = {m, w, h, w};
        return true;
    }

private:
    Device& device_;
    std::vector<void*> buffers_;
};

class CudaDevice final : public Device {
public:
    static std::unique_ptr<CudaDevice> create() {
        auto d = std::unique_ptr<CudaDevice>(new CudaDevice());
        return d->init() ? std::move(d) : nullptr;
    }
    ~CudaDevice() override {
        backend_.reset();
        if (ctx_) destroy_(ctx_);
    }
    const char* name() const override { return "CUDA"; }
    reception::gpu::Backend& backend() override { return *backend_; }
    void* alloc(size_t bytes) override {
        unsigned long long p = 0;
        return alloc_(&p, bytes) == 0 ? reinterpret_cast<void*>(p) : nullptr;
    }
    void release(void* mem) override { free_(reinterpret_cast<unsigned long long>(mem)); }
    bool upload(void* mem, const void* d, size_t bytes) override {
        return htod_(reinterpret_cast<unsigned long long>(mem), d, bytes) == 0;
    }
    bool download(void* mem, void* d, size_t bytes) override {
        return sync_() == 0 && dtoh_(d, reinterpret_cast<unsigned long long>(mem), bytes) == 0;
    }
    bool finish() override { return sync_() == 0; }

private:
    bool init() {
        HMODULE dll = LoadLibraryA("nvcuda.dll");
        if (!dll) return false;
        int (*cu_init)(unsigned) = nullptr;
        int (*device_get)(int*, int) = nullptr;
        int (*ctx_create)(void**, unsigned, int) = nullptr;
        if (!load_fn(dll, cu_init, "cuInit") || !load_fn(dll, device_get, "cuDeviceGet") ||
            !load_fn(dll, ctx_create, "cuCtxCreate_v2") || !load_fn(dll, destroy_, "cuCtxDestroy_v2") ||
            !load_fn(dll, alloc_, "cuMemAlloc_v2") || !load_fn(dll, free_, "cuMemFree_v2") ||
            !load_fn(dll, htod_, "cuMemcpyHtoD_v2") || !load_fn(dll, dtoh_, "cuMemcpyDtoH_v2") ||
            !load_fn(dll, sync_, "cuCtxSynchronize"))
            return false;
        int dev = 0;
        if (cu_init(0) != 0 || device_get(&dev, 0) != 0 || ctx_create(&ctx_, 0, dev) != 0) return false;
        backend_ = reception::gpu::make_cuda(ctx_, nullptr);
        return backend_ != nullptr;
    }

    void* ctx_ = nullptr;
    std::unique_ptr<reception::gpu::Backend> backend_;
    int (*destroy_)(void*) = nullptr;
    int (*alloc_)(unsigned long long*, size_t) = nullptr;
    int (*free_)(unsigned long long) = nullptr;
    int (*htod_)(unsigned long long, const void*, size_t) = nullptr;
    int (*dtoh_)(void*, unsigned long long, size_t) = nullptr;
    int (*sync_)() = nullptr;
};

class OpenCLDevice final : public Device {
public:
    static std::unique_ptr<OpenCLDevice> create() {
        auto d = std::unique_ptr<OpenCLDevice>(new OpenCLDevice());
        return d->init() ? std::move(d) : nullptr;
    }
    ~OpenCLDevice() override {
        backend_.reset();
        if (queue_) release_queue_(queue_);
        if (ctx_) release_ctx_(ctx_);
    }
    const char* name() const override { return label_.c_str(); }
    reception::gpu::Backend& backend() override { return *backend_; }
    void* alloc(size_t bytes) override {
        int err = 0;
        void* m = create_buffer_(ctx_, 1 /* CL_MEM_READ_WRITE */, bytes, nullptr, &err);
        return err == 0 ? m : nullptr;
    }
    void release(void* mem) override { release_mem_(mem); }
    bool upload(void* mem, const void* d, size_t bytes) override {
        return write_(queue_, mem, 1, 0, bytes, d, 0, nullptr, nullptr) == 0;
    }
    bool download(void* mem, void* d, size_t bytes) override {
        return read_(queue_, mem, 1, 0, bytes, d, 0, nullptr, nullptr) == 0;
    }
    bool finish() override { return finish_(queue_) == 0; }

private:
    bool init() {
        HMODULE dll = LoadLibraryA("OpenCL.dll");
        if (!dll) return false;
        int(__stdcall * get_platforms)(unsigned, void**, unsigned*) = nullptr;
        int(__stdcall * get_devices)(void*, unsigned long long, unsigned, void**, unsigned*) = nullptr;
        int(__stdcall * get_platform_info)(void*, unsigned, size_t, void*, size_t*) = nullptr;
        void*(__stdcall * create_ctx)(const void*, unsigned, void**, void*, void*, int*) = nullptr;
        void*(__stdcall * create_queue)(void*, void*, unsigned long long, int*) = nullptr;
        if (!load_fn(dll, get_platforms, "clGetPlatformIDs") || !load_fn(dll, get_devices, "clGetDeviceIDs") ||
            !load_fn(dll, get_platform_info, "clGetPlatformInfo") ||
            !load_fn(dll, create_ctx, "clCreateContext") || !load_fn(dll, create_queue, "clCreateCommandQueue") ||
            !load_fn(dll, create_buffer_, "clCreateBuffer") || !load_fn(dll, write_, "clEnqueueWriteBuffer") ||
            !load_fn(dll, read_, "clEnqueueReadBuffer") || !load_fn(dll, release_mem_, "clReleaseMemObject") ||
            !load_fn(dll, release_queue_, "clReleaseCommandQueue") || !load_fn(dll, release_ctx_, "clReleaseContext") ||
            !load_fn(dll, finish_, "clFinish"))
            return false;
        void* platforms[8] = {};
        unsigned count = 0;
        if (get_platforms(8, platforms, &count) != 0) return false;
        for (unsigned i = 0; i < count && !backend_; ++i) {
            void* device = nullptr;
            unsigned n = 0;
            if (get_devices(platforms[i], 1ull << 2 /* CL_DEVICE_TYPE_GPU */, 1, &device, &n) != 0 || n == 0)
                continue;
            int err = 0;
            ctx_ = create_ctx(nullptr, 1, &device, nullptr, nullptr, &err);
            if (err != 0) { ctx_ = nullptr; continue; }
            queue_ = create_queue(ctx_, device, 0, &err);
            char vendor[128] = {};
            get_platform_info(platforms[i], 0x0903 /* CL_PLATFORM_NAME */, sizeof(vendor), vendor, nullptr);
            label_ = std::string("OpenCL (") + vendor + ")";
            backend_ = reception::gpu::make_opencl(ctx_, device, queue_);
        }
        return backend_ != nullptr;
    }

    void* ctx_ = nullptr;
    void* queue_ = nullptr;
    std::string label_ = "OpenCL";
    std::unique_ptr<reception::gpu::Backend> backend_;
    void*(__stdcall* create_buffer_)(void*, unsigned long long, size_t, void*, int*) = nullptr;
    int(__stdcall* write_)(void*, void*, unsigned, size_t, size_t, const void*, unsigned, const void*, void*) = nullptr;
    int(__stdcall* read_)(void*, void*, unsigned, size_t, size_t, void*, unsigned, const void*, void*) = nullptr;
    int(__stdcall* release_mem_)(void*) = nullptr;
    int(__stdcall* release_queue_)(void*) = nullptr;
    int(__stdcall* release_ctx_)(void*) = nullptr;
    int(__stdcall* finish_)(void*) = nullptr;
};

// BGRA premultiplied test pattern: gradients, saturated patches, fine detail,
// transparent border.
std::vector<float4> test_image(int w, int h) {
    std::vector<float4> img(size_t(w) * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float r = float(x) / w, g = float(y) / h, b = 0.5f + 0.5f * std::sin(x * 0.3f);
            if (((x / 8) + (y / 8)) % 2 == 0 && x > w / 2) r = g = b = 0.9f;
            if (x > w / 6 && x < w / 3 && y > h / 4 && y < h / 2) r = 0.9f, g = 0.1f, b = 0.1f;
            const float a = (x < 4 || y < 4) ? 0.0f : 1.0f;
            img[size_t(y) * w + x] = {b * a, g * a, r * a, a};
        }
    return img;
}

struct Scenario {
    const char* name;
    std::function<void(reception::Params&)> setup;
};

bool run(Device& dev, const Scenario& sc, int bits, double ds) {
    const int fw = 320, fh = 180;
    reception::Frame f;
    f.full_w = fw;
    f.full_h = fh;
    f.canvas_w = int(fw * ds);
    f.canvas_h = int(fh * ds);
    f.ds_x = f.ds_y = ds;
    f.bits = bits;
    // Input smaller than the layer and offset; output a sub-rect.
    f.in_ox = 3, f.in_oy = 2, f.in_w = f.canvas_w - 6, f.in_h = f.canvas_h - 4;
    f.out_ox = 5, f.out_oy = 3, f.out_w = f.canvas_w - 15, f.out_h = f.canvas_h - 10;
    const std::vector<float4> in = test_image(f.in_w, f.in_h);

    reception::Params c;
    sc.setup(c);

    void* in_mem = dev.alloc(in.size() * 16);
    void* out_mem = dev.alloc(size_t(f.out_w) * f.out_h * 16);
    if (!in_mem || !out_mem || !dev.upload(in_mem, in.data(), in.size() * 16)) {
        std::printf("  %-34s alloc/upload failed\n", sc.name);
        return false;
    }
    double max_diff = 0, sum_diff = 0;
    long long samples = 0, outliers = 0;
    std::vector<float4> cpu(size_t(f.out_w) * f.out_h), gpu(cpu.size());
    for (int frame = 0; frame < 6; ++frame) {
        f.time = (frame * 7 + 3) * 1001;
        f.time_step = 1001;
        f.time_scale = 30000;
        const ReceptionParams p = reception::make_params(c, f);
        reception::render_cpu(p, in.data(), cpu.data());
        DeviceScratch scratch(dev);
        if (!reception::gpu::render(dev.backend(), scratch, p, {in_mem, f.in_w, f.in_h, f.in_w},
                                 {out_mem, f.out_w, f.out_h, f.out_w}) ||
            !dev.download(out_mem, gpu.data(), gpu.size() * 16)) {
            std::printf("  %-34s GPU render failed\n", sc.name);
            return false;
        }
        for (size_t i = 0; i < cpu.size(); ++i) {
            const float d[4] = {cpu[i].x - gpu[i].x, cpu[i].y - gpu[i].y, cpu[i].z - gpu[i].z, cpu[i].w - gpu[i].w};
            for (float v : d) {
                const double a = std::abs(double(v));
                if (!(a == a)) { max_diff = 1e9; ++outliers; continue; }
                max_diff = std::max(max_diff, a);
                sum_diff += a;
                outliers += a > 0.02;
                ++samples;
            }
        }
    }
    dev.release(in_mem);
    dev.release(out_mem);
    const double mean = sum_diff / double(samples);
    const double share = double(outliers) / double(samples);
    const bool ok = mean < 2e-3 && share < 5e-3;
    std::printf("  %-34s %s  mean %.6f  max %.4f  outliers %.3f%%\n", sc.name, ok ? "PASS" : "FAIL", mean, max_diff,
                share * 100);
    return ok;
}

void bench(Device& dev) {
    const int w = 1920, h = 1080;
    reception::Frame f;
    f.canvas_w = f.in_w = f.out_w = w;
    f.canvas_h = f.in_h = f.out_h = h;
    f.full_w = w;
    f.full_h = h;
    f.time_scale = 30;
    const std::vector<float4> in = test_image(w, h);
    const reception::Params c;
    void* in_mem = dev.alloc(in.size() * 16);
    void* out_mem = dev.alloc(in.size() * 16);
    dev.upload(in_mem, in.data(), in.size() * 16);
    double best = 1e9;
    for (int i = 0; i < 8; ++i) {
        f.time = i;
        f.time_step = 1;
        const ReceptionParams p = reception::make_params(c, f);
        DeviceScratch scratch(dev);
        dev.finish();
        const auto t0 = std::chrono::steady_clock::now();
        reception::gpu::render(dev.backend(), scratch, p, {in_mem, w, h, w}, {out_mem, w, h, w});
        dev.finish();
        best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::printf("  1080p defaults: %.1f ms per frame\n", best);
    dev.release(in_mem);
    dev.release(out_mem);
}

void bench_cpu() {
    const int w = 1920, h = 1080;
    reception::Frame f;
    f.canvas_w = f.in_w = f.out_w = w;
    f.canvas_h = f.in_h = f.out_h = h;
    f.full_w = w;
    f.full_h = h;
    f.time_scale = 30;
    const std::vector<float4> in = test_image(w, h);
    std::vector<float4> out(in.size());
    double best = 1e9;
    for (int i = 0; i < 4; ++i) {
        f.time = i;
        f.time_step = 1;
        const ReceptionParams p = reception::make_params(reception::Params{}, f);
        const auto t0 = std::chrono::steady_clock::now();
        reception::render_cpu(p, in.data(), out.data());
        best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::printf("CPU, 1080p defaults: %.1f ms per frame\n", best);
}

}  // namespace

int main() {
    const Scenario scenarios[] = {
        {"defaults", [](reception::Params&) {}},
        {"contrast walk, strong noise, mirror",
         [](reception::Params& c) {
             c.edges = reception::Edges::Mirror;
             c.contrast = 30;
             c.contrast_var = 40;
             c.noise_max = 80;
             c.noise_width = 0;
         }},
        {"negative contrast, wide glow",
         [](reception::Params& c) {
             c.contrast = -40;
             c.glow_size = 6;
             c.glow_threshold = 40;
             c.glow_intensity = 3;
         }},
        {"big distortion, transparent edges",
         [](reception::Params& c) {
             c.edges = reception::Edges::Transparent;
             c.amp1 = 100;
             c.amp2 = 60;
             c.offset_var = 8;
             c.freq1 = 9;
         }},
        {"ghosts far apart, black edges",
         [](reception::Params& c) {
             c.edges = reception::Edges::Black;
             c.ghost_off2 = 60;
             c.ghost_off3 = 90;
             c.w_ghost1 = 30;
             c.w_ghost3 = 40;
         }},
        {"rolling picture, dark bars, blend 30",
         [](reception::Params& c) {
             c.roll_prob = 100;
             c.roll_speed = 40;
             c.roll_length = 1000;
             c.bar_dark = 100;
             c.bar_height = 80;
             c.blend = 30;
         }},
        {"everything off",
         [](reception::Params& c) {
             c.bright_var = 0;
             c.noise_max = c.micro = 0;
             c.dist_min = c.dist_max = 0;
             c.glow_size = 0;
             c.w_ghost1 = c.w_ghost2 = c.w_ghost3 = 0;
             c.bar_dark = c.bar_height = 0;
             c.roll_prob = 0;
         }},
    };

    bench_cpu();
    bool all_ok = true, any_device = false;
    std::unique_ptr<Device> devices[] = {CudaDevice::create(), OpenCLDevice::create()};
    for (auto& dev : devices) {
        if (!dev) continue;
        any_device = true;
        for (double ds : {1.0, 0.5}) {
            for (int bits : {8, 32}) {
                std::printf("%s, %d bpc, resolution %.2f\n", dev->name(), bits, ds);
                for (const Scenario& sc : scenarios) all_ok &= run(*dev, sc, bits, ds);
            }
        }
        std::printf("%s timing\n", dev->name());
        bench(*dev);
    }
    if (!any_device) {
        std::printf("No CUDA or OpenCL GPU available.\n");
        return 2;
    }
    std::printf(all_ok ? "All GPU checks passed\n" : "Some GPU checks failed\n");
    return all_ok ? 0 : 1;
}
