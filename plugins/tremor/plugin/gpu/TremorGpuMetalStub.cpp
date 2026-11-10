// TremorGpuMetalStub.cpp — Metal only exists on macOS (see TremorGpuMetal.mm).
#include "TremorGpu.h"

namespace tremor::gpu {
std::unique_ptr<Backend> make_metal(void*, void*) { return nullptr; }
}  // namespace tremor::gpu
