// AnalogGpuMetalStub.cpp — Metal only exists on macOS (see AnalogGpuMetal.mm).
#include "AnalogGpu.h"

namespace analog::gpu {
std::unique_ptr<Backend> make_metal(void*, void*) { return nullptr; }
}  // namespace analog::gpu
