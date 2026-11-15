// AnalogGpuCuda.cpp — CUDA backend through the driver API.
//
// The kernels are compiled to PTX at build time and JIT-compiled by the NVIDIA
// driver on first use, so the plugin needs no CUDA runtime DLL: the driver
// (nvcuda.dll, already loaded by After Effects) is enough.
#include "AnalogGpu.h"

#if defined(_WIN32) && defined(ANALOG_HAS_CUDA)

#include <windows.h>

#include <type_traits>

#include "analog_kernels_ptx.h"  // generated: kAnalogPtx

namespace analog::gpu {
namespace {

using CUresult = int;
using CUmodule = void*;
using CUfunction = void*;
using CUcontext = void*;
using CUstream = void*;
constexpr CUresult CUDA_SUCCESS = 0;

struct DriverApi {
    CUresult (*module_load_data)(CUmodule*, const void*) = nullptr;
    CUresult (*module_get_function)(CUfunction*, CUmodule, const char*) = nullptr;
    CUresult (*module_unload)(CUmodule) = nullptr;
    CUresult (*launch_kernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                              unsigned, CUstream, void**, void**) = nullptr;
    CUresult (*ctx_push)(CUcontext) = nullptr;
    CUresult (*ctx_pop)(CUcontext*) = nullptr;

    bool load() {
        HMODULE dll = GetModuleHandleA("nvcuda.dll");
        if (!dll) dll = LoadLibraryA("nvcuda.dll");
        if (!dll) return false;
        auto get = [dll](auto& fn, const char* name) {
            fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(GetProcAddress(dll, name));
            return fn != nullptr;
        };
        return get(module_load_data, "cuModuleLoadData") &&
               get(module_get_function, "cuModuleGetFunction") &&
               get(module_unload, "cuModuleUnload") && get(launch_kernel, "cuLaunchKernel") &&
               get(ctx_push, "cuCtxPushCurrent_v2") && get(ctx_pop, "cuCtxPopCurrent_v2");
    }
};

// Makes the host's context current for the lifetime of the scope.
class ContextScope {
public:
    ContextScope(const DriverApi& api, CUcontext ctx) : api_(api), ok_(api.ctx_push(ctx) == CUDA_SUCCESS) {}
    ~ContextScope() {
        CUcontext ignored = nullptr;
        if (ok_) api_.ctx_pop(&ignored);
    }
    bool ok() const { return ok_; }

private:
    const DriverApi& api_;
    bool ok_;
};

class CudaBackend final : public Backend {
public:
    CudaBackend(const DriverApi& api, CUcontext ctx, CUstream stream) : api_(api), ctx_(ctx), stream_(stream) {}

    ~CudaBackend() override {
        if (!module_) return;
        ContextScope scope(api_, ctx_);
        api_.module_unload(module_);
    }

    bool init() {
        ContextScope scope(api_, ctx_);
        if (!scope.ok() || api_.module_load_data(&module_, kAnalogPtx) != CUDA_SUCCESS) return false;
        for (int k = 0; k < K_COUNT; ++k)
            if (api_.module_get_function(&fn_[k], module_, kernel_name(Kernel(k))) != CUDA_SUCCESS) return false;
        return true;
    }

    bool run(Kernel k, void* a, void* b, void* c, const AnalogParams& p, int w, int h) override {
        ContextScope scope(api_, ctx_);
        if (!scope.ok() || w <= 0 || h <= 0) return scope.ok();
        AnalogParams params = p;
        void* args3[] = {&a, &b, &c, &params};
        void* args2[] = {&a, &b, &params};
        // Per-row kernels (h == 1) use wide 1-D blocks; image kernels 16 x 16 tiles.
        const unsigned bx = h == 1 ? 64u : 16u, by = h == 1 ? 1u : 16u;
        const unsigned gx = (unsigned(w) + bx - 1) / bx, gy = (unsigned(h) + by - 1) / by;
        return api_.launch_kernel(fn_[k], gx, gy, 1, bx, by, 1, 0, stream_, three_buffers(k) ? args3 : args2,
                                  nullptr) == CUDA_SUCCESS;
    }

private:
    const DriverApi& api_;
    CUcontext ctx_;
    CUstream stream_;
    CUmodule module_ = nullptr;
    CUfunction fn_[K_COUNT] = {};
};

}  // namespace

std::unique_ptr<Backend> make_cuda(void* context, void* stream) {
    static DriverApi api;
    static const bool loaded = api.load();
    if (!loaded || !context) return nullptr;
    auto backend = std::make_unique<CudaBackend>(api, context, stream);
    if (!backend->init()) return nullptr;
    return backend;
}

}  // namespace analog::gpu

#else

namespace analog::gpu {
std::unique_ptr<Backend> make_cuda(void*, void*) { return nullptr; }
}  // namespace analog::gpu

#endif
