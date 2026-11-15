# Analog — added by the root CMakeLists.txt (shared settings and AE_SDK_DIR live there).

add_library(analog_core INTERFACE)
target_include_directories(analog_core INTERFACE core/include)

add_executable(analog_render_frames tools/render_frames.cpp)
target_link_libraries(analog_render_frames PRIVATE analog_core)
if(MSVC)
  target_compile_definitions(analog_render_frames PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

add_executable(analog_test_core tests/test_core.cpp)
target_link_libraries(analog_test_core PRIVATE analog_core)
add_test(NAME analog_core COMMAND analog_test_core)

# --- GPU kernels and backends ------------------------------------------------
# One source (core/include/analog_kernels.inc) feeds the CPU renderer and three
# GPU backends: OpenCL and Metal compile it at runtime from embedded text; CUDA
# embeds PTX built here with nvcc (optional: without nvcc, CUDA projects render
# on the CPU).
set(core_dir "${CMAKE_CURRENT_SOURCE_DIR}/core/include")
set(gpu_dir "${CMAKE_CURRENT_SOURCE_DIR}/plugin/gpu")
set(gpu_gen "${CMAKE_CURRENT_BINARY_DIR}/gpu")
file(MAKE_DIRECTORY "${gpu_gen}")
set(kernel_inputs "${core_dir}/analog_types.h" "${core_dir}/analog_kernels.inc")

function(analog_embed output name)
  list(JOIN ARGN "|" files)
  add_custom_command(
    OUTPUT "${output}"
    COMMAND "${CMAKE_COMMAND}" "-DOUTPUT=${output}" "-DNAME=${name}" "-DFILES=${files}"
            -P "${gpu_dir}/embed.cmake"
    DEPENDS ${ARGN} "${gpu_dir}/embed.cmake"
    COMMENT "Embedding ${name}"
    VERBATIM)
endfunction()

set(gpu_sources "${gpu_dir}/AnalogGpuCuda.cpp" "${gpu_dir}/AnalogGpuOpenCL.cpp")
set(gpu_defs "")
if(WIN32)
  analog_embed("${gpu_gen}/analog_kernels_opencl.h" kAnalogOpenCLSource
               "${gpu_dir}/prelude_opencl.cl" ${kernel_inputs})
  list(APPEND gpu_sources "${gpu_gen}/analog_kernels_opencl.h" "${gpu_dir}/AnalogGpuMetalStub.cpp")

  find_program(ANALOG_NVCC nvcc
               HINTS "$ENV{CUDA_PATH}/bin" "$ENV{USERPROFILE}/Tools/cuda-12.8/bin")
  if(ANALOG_NVCC)
    get_filename_component(host_cl_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
    add_custom_command(
      OUTPUT "${gpu_gen}/analog_kernels.ptx"
      COMMAND "${ANALOG_NVCC}" -ptx -arch=compute_50 -Wno-deprecated-gpu-targets
              -allow-unsupported-compiler -ccbin "${host_cl_dir}" -I "${core_dir}"
              "${gpu_dir}/analog_kernels.cu" -o "${gpu_gen}/analog_kernels.ptx"
      DEPENDS "${gpu_dir}/analog_kernels.cu" ${kernel_inputs}
      COMMENT "Compiling CUDA kernels to PTX"
      VERBATIM)
    analog_embed("${gpu_gen}/analog_kernels_ptx.h" kAnalogPtx "${gpu_gen}/analog_kernels.ptx")
    list(APPEND gpu_sources "${gpu_gen}/analog_kernels_ptx.h")
    list(APPEND gpu_defs ANALOG_HAS_CUDA)
    message(STATUS "Analog CUDA kernels: ${ANALOG_NVCC}")
  else()
    message(STATUS "nvcc not found: CUDA projects will render Analog on the CPU")
  endif()
elseif(APPLE)
  enable_language(OBJCXX)
  analog_embed("${gpu_gen}/analog_kernels_metal.h" kAnalogMetalSource
               "${gpu_dir}/prelude_metal.metal" ${kernel_inputs})
  list(APPEND gpu_sources "${gpu_dir}/AnalogGpuMetal.mm" "${gpu_gen}/analog_kernels_metal.h")
  set_source_files_properties("${gpu_dir}/AnalogGpuMetal.mm" PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
else()
  return()
endif()

add_library(analog_gpu STATIC ${gpu_sources})
target_include_directories(analog_gpu PUBLIC "${gpu_dir}" "${gpu_gen}")
target_compile_definitions(analog_gpu PRIVATE ${gpu_defs})
target_link_libraries(analog_gpu PUBLIC analog_core)
if(WIN32)
  target_compile_definitions(analog_gpu PUBLIC NOMINMAX PRIVATE WIN32_LEAN_AND_MEAN)
  # Runs the GPU kernels on this machine (CUDA and OpenCL) and compares them with
  # the CPU renderer. Not part of ctest because CI has no GPU.
  add_executable(analog_gpu_check tools/gpu_check.cpp)
  target_link_libraries(analog_gpu_check PRIVATE analog_gpu)
else()
  target_link_libraries(analog_gpu PUBLIC "-framework Metal" "-framework Foundation")
endif()

# --- After Effects plugin --------------------------------------------------
if(NOT EXISTS "${AE_SDK_DIR}/Headers/AE_Effect.h")
  message(STATUS "AE SDK not found in ${AE_SDK_DIR}: skipping the Analog plugin")
  return()
endif()

set(pipl_src "${CMAKE_CURRENT_SOURCE_DIR}/plugin/AnalogPiPL.r")
set(plugin_sources
  plugin/Analog.cpp
  "${AE_SDK_DIR}/Util/AEGP_SuiteHandler.cpp"
  "${AE_SDK_DIR}/Util/MissingSuiteError.cpp")

if(WIN32)
  set(pipl_rc "${CMAKE_CURRENT_BINARY_DIR}/AnalogPiPL.rc")
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
    COMMENT "Compiling the Analog PiPL"
    VERBATIM)

  add_library(Analog MODULE ${plugin_sources} "${pipl_rc}")
  set_target_properties(Analog PROPERTIES SUFFIX ".aex" PREFIX "")
  target_compile_definitions(Analog PRIVATE MSWindows WIN32 _WINDOWS NOMINMAX)
elseif(APPLE)
  # The PiPL ships as a data-fork resource in Contents/Resources/Analog.rsrc.
  execute_process(COMMAND xcrun --show-sdk-path OUTPUT_VARIABLE macos_sdk
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(pipl_rsrc "${CMAKE_CURRENT_BINARY_DIR}/Analog.rsrc")
  add_custom_command(
    OUTPUT "${pipl_rsrc}"
    COMMAND xcrun Rez -o "${pipl_rsrc}" -define __MACH__ -useDF -script Roman
            -i "${AE_SDK_DIR}/Headers" -i "${AE_SDK_DIR}/Resources"
            -isysroot "${macos_sdk}" "${pipl_src}"
    DEPENDS "${pipl_src}"
    COMMENT "Compiling the Analog PiPL (Rez)"
    VERBATIM)
  set_source_files_properties("${pipl_rsrc}" PROPERTIES MACOSX_PACKAGE_LOCATION Resources)

  add_library(Analog MODULE ${plugin_sources} "${pipl_rsrc}")
  set_target_properties(Analog PROPERTIES
    BUNDLE TRUE
    BUNDLE_EXTENSION plugin
    OUTPUT_NAME Analog
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/plugin/Info.plist.in"
    MACOSX_BUNDLE_GUI_IDENTIFIER "com.xjfx.Analog"
    CXX_VISIBILITY_PRESET hidden)
endif()

target_include_directories(Analog PRIVATE
  "${AE_SDK_DIR}/Headers" "${AE_SDK_DIR}/Headers/SP"
  "${AE_SDK_DIR}/Resources" "${AE_SDK_DIR}/Util")
target_link_libraries(Analog PRIVATE analog_gpu)
