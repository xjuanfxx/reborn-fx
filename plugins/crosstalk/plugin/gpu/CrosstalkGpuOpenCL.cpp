// CrosstalkGpuOpenCL.cpp — OpenCL backend (Windows; used by AMD and Intel GPUs, or
// NVIDIA when the project is set to OpenCL).
//
// OpenCL.dll is loaded at runtime (After Effects already has it loaded when it
// renders with OpenCL), so building needs no OpenCL SDK. The kernels are
// compiled from source when the device is set up.
#include "CrosstalkGpu.h"

#ifdef _WIN32

#include <windows.h>

#include <mutex>
#include <type_traits>

#include "crosstalk_kernels_opencl.h"  // generated: kCrosstalkOpenCLSource

namespace crosstalk::gpu {
namespace {

using cl_int = int;
using cl_uint = unsigned;
using cl_handle = void*;  // cl_context, cl_device_id, cl_command_queue, cl_program, cl_kernel, cl_mem
constexpr cl_int CL_SUCCESS = 0;

struct OpenCLApi {
    cl_handle(__stdcall* create_program)(cl_handle, cl_uint, const char**, const size_t*, cl_int*) = nullptr;
    cl_int(__stdcall* build_program)(cl_handle, cl_uint, const cl_handle*, const char*, void*, void*) = nullptr;
    cl_handle(__stdcall* create_kernel)(cl_handle, const char*, cl_int*) = nullptr;
    cl_int(__stdcall* set_arg)(cl_handle, cl_uint, size_t, const void*) = nullptr;
    cl_int(__stdcall* enqueue)(cl_handle, cl_handle, cl_uint, const size_t*, const size_t*, const size_t*,
                               cl_uint, const void*, void*) = nullptr;
    cl_int(__stdcall* release_kernel)(cl_handle) = nullptr;
    cl_int(__stdcall* release_program)(cl_handle) = nullptr;

    bool load() {
        HMODULE dll = GetModuleHandleA("OpenCL.dll");
        if (!dll) dll = LoadLibraryA("OpenCL.dll");
        if (!dll) return false;
        auto get = [dll](auto& fn, const char* name) {
            fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(GetProcAddress(dll, name));
            return fn != nullptr;
        };
        return get(create_program, "clCreateProgramWithSource") && get(build_program, "clBuildProgram") &&
               get(create_kernel, "clCreateKernel") && get(set_arg, "clSetKernelArg") &&
               get(enqueue, "clEnqueueNDRangeKernel") && get(release_kernel, "clReleaseKernel") &&
               get(release_program, "clReleaseProgram");
    }
};

class OpenCLBackend final : public Backend {
public:
    OpenCLBackend(const OpenCLApi& api, cl_handle ctx, cl_handle device, cl_handle queue)
        : api_(api), ctx_(ctx), device_(device), queue_(queue) {}

    ~OpenCLBackend() override {
        for (cl_handle k : kernels_)
            if (k) api_.release_kernel(k);
        if (program_) api_.release_program(program_);
    }

    bool init() {
        cl_int err = CL_SUCCESS;
        const char* source = kCrosstalkOpenCLSource;
        program_ = api_.create_program(ctx_, 1, &source, nullptr, &err);
        if (err != CL_SUCCESS || !program_) return false;
        // No fast-relaxed-math, and IEEE division: rows land on whole pixels, so the
        // displacement must round like the CPU renderer.
        if (api_.build_program(program_, 1, &device_, "-cl-fp32-correctly-rounded-divide-sqrt", nullptr, nullptr) !=
            CL_SUCCESS)
            return false;
        for (int k = 0; k < K_COUNT; ++k) {
            kernels_[k] = api_.create_kernel(program_, kernel_name(Kernel(k)), &err);
            if (err != CL_SUCCESS) return false;
        }
        return true;
    }

    bool run(Kernel k, void* a, void* b, void* c, void* dst, const CrosstalkParams& p, int w, int h) override {
        if (w <= 0 || h <= 0) return true;
        std::lock_guard<std::mutex> lock(mutex_);
        cl_handle kernel = kernels_[k];
        cl_uint i = 0;
        if (!arg(kernel, i++, a) || !arg(kernel, i++, b) || !arg(kernel, i++, c) || !arg(kernel, i++, dst)) return false;
        if (api_.set_arg(kernel, i, sizeof(p), &p) != CL_SUCCESS) return false;
        const size_t bx = h == 1 ? 64 : 16, by = h == 1 ? 1 : 16;
        const size_t global[2] = {(size_t(w) + bx - 1) / bx * bx, (size_t(h) + by - 1) / by * by};
        const size_t local[2] = {bx, by};
        return api_.enqueue(queue_, kernel, 2, nullptr, global, local, 0, nullptr, nullptr) == CL_SUCCESS;
    }

private:
    bool arg(cl_handle kernel, cl_uint index, void* mem) {
        return api_.set_arg(kernel, index, sizeof(cl_handle), &mem) == CL_SUCCESS;
    }

    const OpenCLApi& api_;
    cl_handle ctx_, device_, queue_;
    cl_handle program_ = nullptr;
    cl_handle kernels_[K_COUNT] = {};
    // Kernel arguments are shared state; renders on several threads must not interleave.
    std::mutex mutex_;
};

}  // namespace

std::unique_ptr<Backend> make_opencl(void* context, void* device, void* queue) {
    static OpenCLApi api;
    static const bool loaded = api.load();
    if (!loaded || !context || !device || !queue) return nullptr;
    auto backend = std::make_unique<OpenCLBackend>(api, context, device, queue);
    if (!backend->init()) return nullptr;
    return backend;
}

}  // namespace crosstalk::gpu

#else

namespace crosstalk::gpu {
std::unique_ptr<Backend> make_opencl(void*, void*, void*) { return nullptr; }
}  // namespace crosstalk::gpu

#endif
