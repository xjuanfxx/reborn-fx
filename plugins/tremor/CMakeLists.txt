# Tremor — added by the root CMakeLists.txt (shared settings and AE_SDK_DIR live there).

add_library(tremor_core INTERFACE)
target_include_directories(tremor_core INTERFACE core/include)

add_executable(render_frames tools/render_frames.cpp)
target_link_libraries(render_frames PRIVATE tremor_core)
if(MSVC)
  target_compile_definitions(render_frames PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

add_executable(test_core tests/test_core.cpp)
target_link_libraries(test_core PRIVATE tremor_core)
add_test(NAME core COMMAND test_core)

add_executable(test_render tests/test_render.cpp)
target_link_libraries(test_render PRIVATE tremor_core)
add_test(NAME render COMMAND test_render)

# --- GPU kernels and backends ------------------------------------------------
# One kernel source (plugin/gpu/tremor_kernels.inc) feeds three backends:
# OpenCL and Metal compile it at runtime from embedded text; CUDA embeds PTX
# built here with nvcc (optional: without nvcc, CUDA projects render on the CPU).
set(gpu_dir "${CMAKE_CURRENT_SOURCE_DIR}/plugin/gpu")
set(gpu_gen "${CMAKE_CURRENT_BINARY_DIR}/gpu")
file(MAKE_DIRECTORY "${gpu_gen}")
set(kernel_inputs "${gpu_dir}/tremor_gpu_types.h" "${gpu_dir}/tremor_kernels.inc")

function(tremor_embed output name)
  list(JOIN ARGN "|" files)
  add_custom_command(
    OUTPUT "${output}"
    COMMAND "${CMAKE_COMMAND}" "-DOUTPUT=${output}" "-DNAME=${name}" "-DFILES=${files}"
            -P "${gpu_dir}/embed.cmake"
    DEPENDS ${ARGN} "${gpu_dir}/embed.cmake"
    COMMENT "Embedding ${name}"
    VERBATIM)
endfunction()

set(gpu_sources "${gpu_dir}/TremorGpuCuda.cpp" "${gpu_dir}/TremorGpuOpenCL.cpp")
set(gpu_defs "")
if(WIN32)
  tremor_embed("${gpu_gen}/tremor_kernels_opencl.h" kTremorOpenCLSource
               "${gpu_dir}/prelude_opencl.cl" ${kernel_inputs})
  list(APPEND gpu_sources "${gpu_gen}/tremor_kernels_opencl.h" "${gpu_dir}/TremorGpuMetalStub.cpp")

  find_program(TREMOR_NVCC nvcc
               HINTS "$ENV{CUDA_PATH}/bin" "$ENV{USERPROFILE}/Tools/cuda-12.8/bin")
  if(TREMOR_NVCC)
    get_filename_component(host_cl_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
    add_custom_command(
      OUTPUT "${gpu_gen}/tremor_kernels.ptx"
      COMMAND "${TREMOR_NVCC}" -ptx -arch=compute_50 -Wno-deprecated-gpu-targets
              -allow-unsupported-compiler -ccbin "${host_cl_dir}" -I "${gpu_dir}"
              "${gpu_dir}/tremor_kernels.cu" -o "${gpu_gen}/tremor_kernels.ptx"
      DEPENDS "${gpu_dir}/tremor_kernels.cu" ${kernel_inputs}
      COMMENT "Compiling CUDA kernels to PTX"
      VERBATIM)
    tremor_embed("${gpu_gen}/tremor_kernels_ptx.h" kTremorPtx "${gpu_gen}/tremor_kernels.ptx")
    list(APPEND gpu_sources "${gpu_gen}/tremor_kernels_ptx.h")
    list(APPEND gpu_defs TREMOR_HAS_CUDA)
    message(STATUS "CUDA kernels: ${TREMOR_NVCC}")
  else()
    message(STATUS "nvcc not found: CUDA projects will render Tremor on the CPU")
  endif()
elseif(APPLE)
  enable_language(OBJCXX)
  tremor_embed("${gpu_gen}/tremor_kernels_metal.h" kTremorMetalSource
               "${gpu_dir}/prelude_metal.metal" ${kernel_inputs})
  list(APPEND gpu_sources "${gpu_dir}/TremorGpuMetal.mm" "${gpu_gen}/tremor_kernels_metal.h")
  set_source_files_properties("${gpu_dir}/TremorGpuMetal.mm" PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
endif()

add_library(tremor_gpu STATIC ${gpu_sources})
target_include_directories(tremor_gpu PUBLIC "${gpu_dir}" "${gpu_gen}")
target_compile_definitions(tremor_gpu PRIVATE ${gpu_defs})
target_link_libraries(tremor_gpu PUBLIC tremor_core)
if(WIN32)
  target_compile_definitions(tremor_gpu PUBLIC NOMINMAX PRIVATE WIN32_LEAN_AND_MEAN)
elseif(APPLE)
  target_link_libraries(tremor_gpu PUBLIC "-framework Metal" "-framework Foundation")
endif()

# Runs the GPU kernels on this machine (CUDA and OpenCL) and compares them with
# the CPU renderer. Windows only; not part of ctest because CI has no GPU.
if(WIN32)
  add_executable(gpu_check tools/gpu_check.cpp)
  target_link_libraries(gpu_check PRIVATE tremor_gpu)
endif()

# --- After Effects plugin --------------------------------------------------
if(NOT EXISTS "${AE_SDK_DIR}/Headers/AE_Effect.h")
  message(STATUS "AE SDK not found in ${AE_SDK_DIR}: skipping the plugin")
  return()
endif()

set(pipl_src "${CMAKE_CURRENT_SOURCE_DIR}/plugin/TremorPiPL.r")
set(plugin_sources
  plugin/Tremor.cpp
  "${AE_SDK_DIR}/Util/AEGP_SuiteHandler.cpp"
  "${AE_SDK_DIR}/Util/MissingSuiteError.cpp")

if(WIN32)
  set(pipl_rc "${CMAKE_CURRENT_BINARY_DIR}/TremorPiPL.rc")
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
    COMMENT "Compiling the PiPL"
    VERBATIM)

  add_library(Tremor MODULE ${plugin_sources} "${pipl_rc}")
  set_target_properties(Tremor PROPERTIES SUFFIX ".aex" PREFIX "")
  target_compile_definitions(Tremor PRIVATE MSWindows WIN32 _WINDOWS NOMINMAX)

elseif(APPLE)
  # The PiPL ships as a data-fork resource in Contents/Resources/Tremor.rsrc.
  execute_process(COMMAND xcrun --show-sdk-path OUTPUT_VARIABLE macos_sdk
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(pipl_rsrc "${CMAKE_CURRENT_BINARY_DIR}/Tremor.rsrc")
  add_custom_command(
    OUTPUT "${pipl_rsrc}"
    COMMAND xcrun Rez -o "${pipl_rsrc}" -define __MACH__ -useDF -script Roman
            -i "${AE_SDK_DIR}/Headers" -i "${AE_SDK_DIR}/Resources"
            -isysroot "${macos_sdk}" "${pipl_src}"
    DEPENDS "${pipl_src}"
    COMMENT "Compiling the PiPL (Rez)"
    VERBATIM)
  set_source_files_properties("${pipl_rsrc}" PROPERTIES MACOSX_PACKAGE_LOCATION Resources)

  add_library(Tremor MODULE ${plugin_sources} "${pipl_rsrc}")
  set_target_properties(Tremor PROPERTIES
    BUNDLE TRUE
    BUNDLE_EXTENSION plugin
    OUTPUT_NAME Tremor
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/plugin/Info.plist.in"
    MACOSX_BUNDLE_GUI_IDENTIFIER "com.xjfx.Tremor"
    CXX_VISIBILITY_PRESET hidden)
else()
  message(STATUS "Platform not supported for the AE plugin")
  return()
endif()

target_include_directories(Tremor PRIVATE
  "${AE_SDK_DIR}/Headers" "${AE_SDK_DIR}/Headers/SP"
  "${AE_SDK_DIR}/Resources" "${AE_SDK_DIR}/Util")
target_link_libraries(Tremor PRIVATE tremor_gpu)
