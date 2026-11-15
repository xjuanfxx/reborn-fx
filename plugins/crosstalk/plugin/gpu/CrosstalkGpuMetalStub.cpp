// CrosstalkGpuMetalStub.cpp — Metal only exists on macOS (see CrosstalkGpuMetal.mm).
#include "CrosstalkGpu.h"

namespace crosstalk::gpu {
std::unique_ptr<Backend> make_metal(void*, void*) { return nullptr; }
}  // namespace crosstalk::gpu
