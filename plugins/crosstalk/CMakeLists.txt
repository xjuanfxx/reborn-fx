# Crosstalk — added by the root CMakeLists.txt (shared settings and AE_SDK_DIR live there).

add_library(crosstalk_core INTERFACE)
target_include_directories(crosstalk_core INTERFACE core/include)

add_executable(crosstalk_test_core tests/test_core.cpp)
target_link_libraries(crosstalk_test_core PRIVATE crosstalk_core)
add_test(NAME crosstalk_core COMMAND crosstalk_test_core)

# --- GPU kernels and backends ------------------------------------------------
# One source (core/include/crosstalk_kernels.inc) feeds the CPU renderer and three
# GPU backends: OpenCL and Metal compile it at runtime from embedded text; CUDA
# embeds PTX built here with nvcc (optional: without nvcc, CUDA projects render
# on the CPU).
set(core_dir "${CMAKE_CURRENT_SOURCE_DIR}/core/include")
set(gpu_dir "${CMAKE_CURRENT_SOURCE_DIR}/plugin/gpu")
set(gpu_gen "${CMAKE_CURRENT_BINARY_DIR}/gpu")
file(MAKE_DIRECTORY "${gpu_gen}")
set(kernel_inputs "${core_dir}/crosstalk_types.h" "${core_dir}/crosstalk_kernels.inc")

function(crosstalk_embed output name)
  list(JOIN ARGN "|" files)
  add_custom_command(
    OUTPUT "${output}"
    COMMAND "${CMAKE_COMMAND}" "-DOUTPUT=${output}" "-DNAME=${name}" "-DFILES=${files}"
            -P "${gpu_dir}/embed.cmake"
    DEPENDS ${ARGN} "${gpu_dir}/embed.cmake"
    COMMENT "Embedding ${name}"
    VERBATIM)
endfunction()

set(gpu_sources "${gpu_dir}/CrosstalkGpuCuda.cpp" "${gpu_dir}/CrosstalkGpuOpenCL.cpp")
set(gpu_defs "")
if(WIN32)
  crosstalk_embed("${gpu_gen}/crosstalk_kernels_opencl.h" kCrosstalkOpenCLSource
               "${gpu_dir}/prelude_opencl.cl" ${kernel_inputs})
  list(APPEND gpu_sources "${gpu_gen}/crosstalk_kernels_opencl.h" "${gpu_dir}/CrosstalkGpuMetalStub.cpp")

  find_program(CROSSTALK_NVCC nvcc
               HINTS "$ENV{CUDA_PATH}/bin" "$ENV{USERPROFILE}/Tools/cuda-12.8/bin")
  if(CROSSTALK_NVCC)
    get_filename_component(host_cl_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
    add_custom_command(
      OUTPUT "${gpu_gen}/crosstalk_kernels.ptx"
      COMMAND "${CROSSTALK_NVCC}" -ptx -arch=compute_50 -Wno-deprecated-gpu-targets
              -allow-unsupported-compiler -ccbin "${host_cl_dir}" -I "${core_dir}"
              "${gpu_dir}/crosstalk_kernels.cu" -o "${gpu_gen}/crosstalk_kernels.ptx"
      DEPENDS "${gpu_dir}/crosstalk_kernels.cu" ${kernel_inputs}
      COMMENT "Compiling CUDA kernels to PTX"
      VERBATIM)
    crosstalk_embed("${gpu_gen}/crosstalk_kernels_ptx.h" kCrosstalkPtx "${gpu_gen}/crosstalk_kernels.ptx")
    list(APPEND gpu_sources "${gpu_gen}/crosstalk_kernels_ptx.h")
    list(APPEND gpu_defs CROSSTALK_HAS_CUDA)
    message(STATUS "Crosstalk CUDA kernels: ${CROSSTALK_NVCC}")
  else()
    message(STATUS "nvcc not found: CUDA projects will render Crosstalk on the CPU")
  endif()
elseif(APPLE)
  enable_language(OBJCXX)
  crosstalk_embed("${gpu_gen}/crosstalk_kernels_metal.h" kCrosstalkMetalSource
               "${gpu_dir}/prelude_metal.metal" ${kernel_inputs})
  list(APPEND gpu_sources "${gpu_dir}/CrosstalkGpuMetal.mm" "${gpu_gen}/crosstalk_kernels_metal.h")
  set_source_files_properties("${gpu_dir}/CrosstalkGpuMetal.mm" PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
else()
  return()
endif()

add_library(crosstalk_gpu STATIC ${gpu_sources})
target_include_directories(crosstalk_gpu PUBLIC "${gpu_dir}" "${gpu_gen}")
target_compile_definitions(crosstalk_gpu PRIVATE ${gpu_defs})
target_link_libraries(crosstalk_gpu PUBLIC crosstalk_core)
if(WIN32)
  target_compile_definitions(crosstalk_gpu PUBLIC NOMINMAX PRIVATE WIN32_LEAN_AND_MEAN)
  # Runs the GPU kernels on this machine (CUDA and OpenCL) and compares them with
  # the CPU renderer. Not part of ctest because CI has no GPU.
  add_executable(crosstalk_gpu_check tools/gpu_check.cpp)
  target_link_libraries(crosstalk_gpu_check PRIVATE crosstalk_gpu)
else()
  target_link_libraries(crosstalk_gpu PUBLIC "-framework Metal" "-framework Foundation")
endif()

# --- After Effects plugin --------------------------------------------------
if(NOT EXISTS "${AE_SDK_DIR}/Headers/AE_Effect.h")
  message(STATUS "AE SDK not found in ${AE_SDK_DIR}: skipping the Crosstalk plugin")
  return()
endif()

set(pipl_src "${CMAKE_CURRENT_SOURCE_DIR}/plugin/CrosstalkPiPL.r")
set(plugin_sources
  plugin/Crosstalk.cpp
  "${AE_SDK_DIR}/Util/AEGP_SuiteHandler.cpp"
  "${AE_SDK_DIR}/Util/MissingSuiteError.cpp")

if(WIN32)
  set(pipl_rc "${CMAKE_CURRENT_BINARY_DIR}/CrosstalkPiPL.rc")
  add_custom_command(
    OUTPUT "${pipl_rc}"
    COMMAND "${CMAKE_COMMAND}"
            "-DCL=${CMAKE_CXX_COMPILER}"
            "-DPIPLTOOL=${AE_SDK_DIR}/Resources/PiPLTool.exe"
            "-DSDK_HEADERS=${AE_SDK_DIR}/Headers"
            "-DINPUT=${pipl_src}"
            "-DOUT_DIR=${CMAKE_CURRENT_BINARY_DIR}"
            "-DOUTPUT=${pipl_rc}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/plugin/pipl.cmake"
    DEPENDS "${pipl_src}" "${CMAKE_CURRENT_SOURCE_DIR}/plugin/pipl.cmake"
    COMMENT "Compiling the Crosstalk PiPL"
    VERBATIM)

  add_library(Crosstalk MODULE ${plugin_sources} "${pipl_rc}")
  set_target_properties(Crosstalk PROPERTIES SUFFIX ".aex" PREFIX "")
  target_compile_definitions(Crosstalk PRIVATE MSWindows WIN32 _WINDOWS NOMINMAX)
elseif(APPLE)
  # The PiPL ships as a data-fork resource in Contents/Resources/Crosstalk.rsrc.
  execute_process(COMMAND xcrun --show-sdk-path OUTPUT_VARIABLE macos_sdk
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(pipl_rsrc "${CMAKE_CURRENT_BINARY_DIR}/Crosstalk.rsrc")
  add_custom_command(
    OUTPUT "${pipl_rsrc}"
    COMMAND xcrun Rez -o "${pipl_rsrc}" -define __MACH__ -useDF -script Roman
            -i "${AE_SDK_DIR}/Headers" -i "${AE_SDK_DIR}/Resources"
            -isysroot "${macos_sdk}" "${pipl_src}"
    DEPENDS "${pipl_src}"
    COMMENT "Compiling the Crosstalk PiPL (Rez)"
    VERBATIM)
  set_source_files_properties("${pipl_rsrc}" PROPERTIES MACOSX_PACKAGE_LOCATION Resources)

  add_library(Crosstalk MODULE ${plugin_sources} "${pipl_rsrc}")
  set_target_properties(Crosstalk PROPERTIES
    BUNDLE TRUE
    BUNDLE_EXTENSION plugin
    OUTPUT_NAME Crosstalk
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/plugin/Info.plist.in"
    MACOSX_BUNDLE_GUI_IDENTIFIER "com.xjfx.Crosstalk"
    CXX_VISIBILITY_PRESET hidden)
endif()

target_include_directories(Crosstalk PRIVATE
  "${AE_SDK_DIR}/Headers" "${AE_SDK_DIR}/Headers/SP"
  "${AE_SDK_DIR}/Resources" "${AE_SDK_DIR}/Util")
target_link_libraries(Crosstalk PRIVATE crosstalk_gpu)
