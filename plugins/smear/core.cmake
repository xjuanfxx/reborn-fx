# Smear — renderer core, its tests and the benchmark. Included by
# plugins/smear/CMakeLists.txt (which adds the GPU backends and the AE plugin);
# needs nothing beyond a C++17 compiler.

set(smear_dir "${CMAKE_CURRENT_LIST_DIR}")

# Header-only: the CPU renderer (core/include) and the GPU pass orchestration
# with its kernel source and parameter blocks (plugin/gpu).
add_library(smear_core INTERFACE)
target_include_directories(smear_core INTERFACE "${smear_dir}/core/include" "${smear_dir}/plugin/gpu")

function(smear_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
    target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
  endif()
endfunction()

add_executable(smear_test_core "${smear_dir}/tests/test_core.cpp")
target_link_libraries(smear_test_core PRIVATE smear_core)
smear_warnings(smear_test_core)
add_test(NAME smear_core COMMAND smear_test_core)

add_executable(smear_test_render "${smear_dir}/tests/test_render.cpp")
target_link_libraries(smear_test_render PRIVATE smear_core)
smear_warnings(smear_test_render)
add_test(NAME smear_render COMMAND smear_test_render)

# The GPU passes (smear_kernels.inc + SmearGpuRender.h) run on the CPU through
# tests/cpu_prelude.h and are compared with the CPU renderer.
add_executable(smear_test_gpu_path "${smear_dir}/tests/test_gpu_path.cpp")
target_link_libraries(smear_test_gpu_path PRIVATE smear_core)
smear_warnings(smear_test_gpu_path)
add_test(NAME smear_gpu_path COMMAND smear_test_gpu_path)

# Quality/speed benchmark on synthetic scenes with known motion (not a test).
add_executable(smear_bench "${smear_dir}/tools/smear_bench.cpp")
target_link_libraries(smear_bench PRIVATE smear_core)
smear_warnings(smear_bench)
