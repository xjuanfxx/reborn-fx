// gpu_check — runs Tremor's GPU kernels on this machine and compares them with
// the CPU renderer, without After Effects.
//
// Creates its own CUDA (driver API) and OpenCL contexts, renders several
// scenarios frame by frame on both paths and reports the pixel difference.
// Static scenarios must match the CPU almost exactly; motion blur scenarios are
// allowed to differ slightly (the CPU uses a box filter along the trail, the GPU
// averages samples).
//
// Usage: gpu_check            (Windows, needs an NVIDIA and/or OpenCL GPU)
#include <windows.h>

#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "TremorGpuRender.h"
#include "tremor_render.h"

namespace {

template <class Fn>
bool load_fn(HMODULE dll, Fn& fn, const char* name) {
    fn = reinterpret_cast<Fn>(GetProcAddress(dll, name));
    return fn != nullptr;
}

// Converts between the CPU image (RGBA) and the GPU layout (BGRA).
std::vector<float> to_bgra(const tremor::Image& img) {
    std::vector<float> v(img.px.size() * 4);
    for (size_t i = 0; i < img.px.size(); ++i) {
        const tremor::Pixel& p = img.px[i];
        v[4 * i] = p.b; v[4 * i + 1] = p.g; v[4 * i + 2] = p.r; v[4 * i + 3] = p.a;
    }
    return v;
}

tremor::Image from_bgra(const std::vector<float>& v, int w, int h) {
    tremor::Image img(w, h);
    for (size_t i = 0; i < img.px.size(); ++i) img.px[i] = {v[4 * i + 2], v[4 * i + 1], v[4 * i], v[4 * i + 3]};
    return img;
}

// A GPU API seen by the test: memory plus a Tremor backend.
class Device {
public:
    virtual ~Device() = default;
    virtual const char* name() const = 0;
    virtual tremor::gpu::Backend& backend() = 0;
    virtual void* alloc(size_t bytes) = 0;
    virtual void release(void* mem) = 0;
    virtual bool upload(void* mem, const std::vector<float>& data) = 0;
    virtual bool download(void* mem, std::vector<float>& data) = 0;
};

class DeviceScratch final : public tremor::gpu::Scratch {
public:
    explicit DeviceScratch(Device& d) : device_(d) {}
    ~DeviceScratch() override {
        for (void* m : buffers_) device_.release(m);
    }
    bool alloc(int w, int h, tremor::gpu::Buffer& out) override {
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

// ---------------------------------------------------------------------------
// CUDA (driver API)
// ---------------------------------------------------------------------------

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
    tremor::gpu::Backend& backend() override { return *backend_; }
    void* alloc(size_t bytes) override {
        unsigned long long p = 0;
        return alloc_(&p, bytes) == 0 ? reinterpret_cast<void*>(p) : nullptr;
    }
    void release(void* mem) override { free_(reinterpret_cast<unsigned long long>(mem)); }
    bool upload(void* mem, const std::vector<float>& d) override {
        return htod_(reinterpret_cast<unsigned long long>(mem), d.data(), d.size() * 4) == 0;
    }
    bool download(void* mem, std::vector<float>& d) override {
        return sync_() == 0 && dtoh_(d.data(), reinterpret_cast<unsigned long long>(mem), d.size() * 4) == 0;
    }

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
        backend_ = tremor::gpu::make_cuda(ctx_, nullptr);
        return backend_ != nullptr;
    }

    void* ctx_ = nullptr;
    std::unique_ptr<tremor::gpu::Backend> backend_;
    int (*destroy_)(void*) = nullptr;
    int (*alloc_)(unsigned long long*, size_t) = nullptr;
    int (*free_)(unsigned long long) = nullptr;
    int (*htod_)(unsigned long long, const void*, size_t) = nullptr;
    int (*dtoh_)(void*, unsigned long long, size_t) = nullptr;
    int (*sync_)() = nullptr;
};

// ---------------------------------------------------------------------------
// OpenCL
// ---------------------------------------------------------------------------

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
    tremor::gpu::Backend& backend() override { return *backend_; }
    void* alloc(size_t bytes) override {
        int err = 0;
        void* m = create_buffer_(ctx_, 1 /* CL_MEM_READ_WRITE */, bytes, nullptr, &err);
        return err == 0 ? m : nullptr;
    }
    void release(void* mem) override { release_mem_(mem); }
    bool upload(void* mem, const std::vector<float>& d) override {
        return write_(queue_, mem, 1, 0, d.size() * 4, d.data(), 0, nullptr, nullptr) == 0;
    }
    bool download(void* mem, std::vector<float>& d) override {
        return read_(queue_, mem, 1, 0, d.size() * 4, d.data(), 0, nullptr, nullptr) == 0;
    }

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
            !load_fn(dll, release_queue_, "clReleaseCommandQueue") || !load_fn(dll, release_ctx_, "clReleaseContext"))
            return false;
        void* platforms[8] = {};
        unsigned count = 0;
        if (get_platforms(8, platforms, &count) != 0) return false;
        for (unsigned i = 0; i < count && !ctx_; ++i) {
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
            backend_ = tremor::gpu::make_opencl(ctx_, device, queue_);
        }
        return backend_ != nullptr;
    }

    void* ctx_ = nullptr;
    void* queue_ = nullptr;
    std::string label_ = "OpenCL";
    std::unique_ptr<tremor::gpu::Backend> backend_;
    void*(__stdcall* create_buffer_)(void*, unsigned long long, size_t, void*, int*) = nullptr;
    int(__stdcall* write_)(void*, void*, unsigned, size_t, size_t, const void*, unsigned, const void*, void*) = nullptr;
    int(__stdcall* read_)(void*, void*, unsigned, size_t, size_t, void*, unsigned, const void*, void*) = nullptr;
    int(__stdcall* release_mem_)(void*) = nullptr;
    int(__stdcall* release_queue_)(void*) = nullptr;
    int(__stdcall* release_ctx_)(void*) = nullptr;
};

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

struct Scenario {
    const char* name;
    std::function<void(tremor::Params&)> setup;
};

tremor::Image test_image(int w, int h) {
    tremor::Image img(w, h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const bool check = ((x / 16) + (y / 16)) % 2 == 0;
            const float a = (x > w / 8 && x < w * 7 / 8) ? 1.0f : 0.0f;  // transparent edges
            img.at(x, y) = {a * float(x) / w, a * (check ? 0.9f : 0.2f), a * float(y) / h, a};
        }
    return img;
}

// Renders `frames` frames on both paths. The input buffer is smaller than the
// layer and offset, and the output is a sub-rect, to exercise the layout.
bool run(Device& dev, const Scenario& sc, int bits) {
    const int w = 320, h = 180, frames = 30;
    const int in_x = 6, in_y = 4, in_w = w - 12, in_h = h - 8;  // input placed at (6,4)
    const int out_x = 10, out_y = 5, out_w = w - 30, out_h = h - 20;
    const tremor::Image full = test_image(w, h);
    tremor::Image in_img(in_w, in_h);
    tremor::Image canvas(w, h);
    for (int y = 0; y < in_h; ++y)
        for (int x = 0; x < in_w; ++x) in_img.at(x, y) = canvas.at(x + in_x, y + in_y) = full.at(x + in_x, y + in_y);
    const std::vector<float> in_bgra = to_bgra(in_img);

    tremor::Params p;
    sc.setup(p);

    void* in_mem = dev.alloc(in_bgra.size() * 4);
    void* out_mem = dev.alloc(size_t(out_w) * out_h * 16);
    if (!in_mem || !out_mem || !dev.upload(in_mem, in_bgra)) {
        std::printf("  %-30s alloc/upload failed\n", sc.name);
        return false;
    }

    double max_diff = 0, sum_diff = 0;
    long long samples = 0, outliers = 0;
    std::vector<float> out_bgra(size_t(out_w) * out_h * 4);
    for (int f = 0; f < frames; ++f) {
        const tremor::FrameState st = tremor::evaluate(p, f * 1001, 30000, w, h);
        const tremor::Look look = tremor::make_look(p, bits);

        tremor::Image cpu(out_w, out_h);
        tremor::render_frame(canvas, cpu, out_x, out_y, st, look);

        tremor::gpu::Layout layout;
        layout.canvas_w = w;
        layout.canvas_h = h;
        layout.in_ox = in_x;
        layout.in_oy = in_y;
        layout.out_ox = out_x;
        layout.out_oy = out_y;
        DeviceScratch scratch(dev);
        if (!tremor::gpu::render(dev.backend(), scratch, st, look, layout, {in_mem, in_w, in_h, in_w},
                                 {out_mem, out_w, out_h, out_w}) ||
            !dev.download(out_mem, out_bgra)) {
            std::printf("  %-30s GPU render failed at frame %d\n", sc.name, f);
            return false;
        }
        const tremor::Image gpu = from_bgra(out_bgra, out_w, out_h);
        for (size_t i = 0; i < cpu.px.size(); ++i) {
            const float d[4] = {cpu.px[i].r - gpu.px[i].r, cpu.px[i].g - gpu.px[i].g, cpu.px[i].b - gpu.px[i].b,
                                cpu.px[i].a - gpu.px[i].a};
            for (float v : d) {
                const double a = std::abs(v);
                max_diff = std::max(max_diff, a);
                sum_diff += a;
                outliers += a > 0.01;
                ++samples;
            }
        }
    }
    dev.release(in_mem);
    dev.release(out_mem);

    // Float rounding can move a sample across a pixel or edge boundary on a
    // handful of pixels; everything else must match closely.
    const double mean = sum_diff / double(samples);
    const double outlier_share = double(outliers) / double(samples);
    const bool ok = mean < 1e-3 && outlier_share < 1e-3;
    std::printf("  %-30s %s  mean %.6f  max %.4f  outliers %.4f%%\n", sc.name, ok ? "PASS" : "FAIL", mean, max_diff,
                outlier_share * 100);
    return ok;
}

}  // namespace

int main() {
    const Scenario scenarios[] = {
        {"defaults", [](tremor::Params&) {}},
        {"blend 100", [](tremor::Params& p) { p.blend = 100; }},
        {"blend 40", [](tremor::Params& p) { p.blend = 40; }},
        {"strong rotation + separation",
         [](tremor::Params& p) {
             p.rot_max = 30;
             p.ch_sep = 40;
         }},
        {"erratic jumps + zoom, mirror",
         [](tremor::Params& p) {
             p.h_erratic = p.v_erratic = 60;
             p.zoom_erratic = 50;
             p.zoom_base = -0.3;
             p.edges = tremor::Edges::Mirror;
         }},
        {"jumps, tile", [](tremor::Params& p) { p.h_erratic = p.v_erratic = 60; p.edges = tremor::Edges::Tile; }},
        {"jumps, black", [](tremor::Params& p) { p.h_erratic = p.v_erratic = 60; p.edges = tremor::Edges::Black; }},
        {"alpha composite + adjust",
         [](tremor::Params& p) {
             p.alpha_composite = true;
             p.h_erratic = 50;
             p.r_adjust = 30;
             p.b_adjust = -20;
         }},
        {"deinterlace lower", [](tremor::Params& p) { p.deinterlace = tremor::Deinterlace::LowerOnly; }},
        {"deinterlace upper", [](tremor::Params& p) { p.deinterlace = tremor::Deinterlace::UpperOnly; }},
        {"deinterlace merge", [](tremor::Params& p) { p.deinterlace = tremor::Deinterlace::Merge; }},
    };

    bool all_ok = true, any_device = false;
    std::unique_ptr<Device> devices[] = {CudaDevice::create(), OpenCLDevice::create()};
    for (auto& dev : devices) {
        if (!dev) continue;
        any_device = true;
        for (int bits : {8, 32}) {
            std::printf("%s, %d bpc\n", dev->name(), bits);
            for (const Scenario& sc : scenarios) all_ok &= run(*dev, sc, bits);
        }
    }
    if (!any_device) {
        std::printf("No CUDA or OpenCL GPU available.\n");
        return 2;
    }
    std::printf(all_ok ? "All GPU checks passed\n" : "Some GPU checks failed\n");
    return all_ok ? 0 : 1;
}
