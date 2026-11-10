// TremorGpuCuda.cpp — CUDA backend through the driver API.
//
// The kernels are compiled to PTX at build time and JIT-compiled by the NVIDIA
// driver on first use, so the plugin needs no CUDA runtime DLL: the driver
// (nvcuda.dll, already loaded by After Effects) is enough.
#include "TremorGpu.h"

#if defined(_WIN32) && defined(TREMOR_HAS_CUDA)

#include <windows.h>

#include <type_traits>

#include "tremor_kernels_ptx.h"  // generated: kTremorPtx

namespace tremor::gpu {
namespace {

using CUresult = int;
using CUmodule = void*;
using CUfunction = void*;
using CUcontext = void*;
using CUstream = void*;
constexpr CUresult CUDA_SUCCESS = 0;
constexpr unsigned kBlock = 16;

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
        return scope.ok() && api_.module_load_data(&module_, kTremorPtx) == CUDA_SUCCESS &&
               api_.module_get_function(&source_, module_, "tremor_source") == CUDA_SUCCESS &&
               api_.module_get_function(&rotate_, module_, "tremor_rotate") == CUDA_SUCCESS &&
               api_.module_get_function(&finish_, module_, "tremor_finish") == CUDA_SUCCESS;
    }

    bool source(void* src, void* dst, const TremorSourceParams& p) override {
        void* args[] = {&src, &dst, const_cast<TremorSourceParams*>(&p)};
        return launch(source_, p.w, p.h, args);
    }
    bool rotate(void* src, void* dst, const TremorRotateParams& p) override {
        void* args[] = {&src, &dst, const_cast<TremorRotateParams*>(&p)};
        return launch(rotate_, p.w, p.h, args);
    }
    bool finish(void* rotated, void* source, void* dst, const TremorFinishParams& p) override {
        void* args[] = {&rotated, &source, &dst, const_cast<TremorFinishParams*>(&p)};
        return launch(finish_, p.out_w, p.out_h, args);
    }

private:
    bool launch(CUfunction fn, int w, int h, void** args) {
        ContextScope scope(api_, ctx_);
        if (!scope.ok()) return false;
        const unsigned gx = (unsigned(w) + kBlock - 1) / kBlock, gy = (unsigned(h) + kBlock - 1) / kBlock;
        return api_.launch_kernel(fn, gx, gy, 1, kBlock, kBlock, 1, 0, stream_, args, nullptr) == CUDA_SUCCESS;
    }

    const DriverApi& api_;
    CUcontext ctx_;
    CUstream stream_;
    CUmodule module_ = nullptr;
    CUfunction source_ = nullptr, rotate_ = nullptr, finish_ = nullptr;
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

}  // namespace tremor::gpu

#else

namespace tremor::gpu {
std::unique_ptr<Backend> make_cuda(void*, void*) { return nullptr; }
}  // namespace tremor::gpu

#endif
