// ReceptionGpuMetalStub.cpp — Metal only exists on macOS (see ReceptionGpuMetal.mm).
#include "ReceptionGpu.h"

namespace reception::gpu {
std::unique_ptr<Backend> make_metal(void*, void*) { return nullptr; }
}  // namespace reception::gpu
