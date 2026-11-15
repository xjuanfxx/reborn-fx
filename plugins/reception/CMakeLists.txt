# Reception — added by the root CMakeLists.txt (shared settings and AE_SDK_DIR live there).

add_library(reception_core INTERFACE)
target_include_directories(reception_core INTERFACE core/include)

add_executable(reception_test_core tests/test_core.cpp)
target_link_libraries(reception_test_core PRIVATE reception_core)
add_test(NAME reception_core COMMAND reception_test_core)

# --- GPU kernels and backends ------------------------------------------------
# One source (core/include/reception_kernels.inc) feeds the CPU renderer and three
# GPU backends: OpenCL and Metal compile it at runtime from embedded text; CUDA
# embeds PTX built here with nvcc (optional: without nvcc, CUDA projects render
# on the CPU).
set(core_dir "${CMAKE_CURRENT_SOURCE_DIR}/core/include")
set(gpu_dir "${CMAKE_CURRENT_SOURCE_DIR}/plugin/gpu")
set(gpu_gen "${CMAKE_CURRENT_BINARY_DIR}/gpu")
file(MAKE_DIRECTORY "${gpu_gen}")
set(kernel_inputs "${core_dir}/reception_types.h" "${core_dir}/reception_kernels.inc")

function(reception_embed output name)
  list(JOIN ARGN "|" files)
  add_custom_command(
    OUTPUT "${output}"
    COMMAND "${CMAKE_COMMAND}" "-DOUTPUT=${output}" "-DNAME=${name}" "-DFILES=${files}"
            -P "${gpu_dir}/embed.cmake"
    DEPENDS ${ARGN} "${gpu_dir}/embed.cmake"
    COMMENT "Embedding ${name}"
    VERBATIM)
endfunction()

set(gpu_sources "${gpu_dir}/ReceptionGpuCuda.cpp" "${gpu_dir}/ReceptionGpuOpenCL.cpp")
set(gpu_defs "")
if(WIN32)
  reception_embed("${gpu_gen}/reception_kernels_opencl.h" kReceptionOpenCLSource
               "${gpu_dir}/prelude_opencl.cl" ${kernel_inputs})
  list(APPEND gpu_sources "${gpu_gen}/reception_kernels_opencl.h" "${gpu_dir}/ReceptionGpuMetalStub.cpp")

  find_program(RECEPTION_NVCC nvcc
               HINTS "$ENV{CUDA_PATH}/bin" "$ENV{USERPROFILE}/Tools/cuda-12.8/bin")
  if(RECEPTION_NVCC)
    get_filename_component(host_cl_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
    add_custom_command(
      OUTPUT "${gpu_gen}/reception_kernels.ptx"
      COMMAND "${RECEPTION_NVCC}" -ptx -arch=compute_50 -Wno-deprecated-gpu-targets
              -allow-unsupported-compiler -ccbin "${host_cl_dir}" -I "${core_dir}"
              "${gpu_dir}/reception_kernels.cu" -o "${gpu_gen}/reception_kernels.ptx"
      DEPENDS "${gpu_dir}/reception_kernels.cu" ${kernel_inputs}
      COMMENT "Compiling CUDA kernels to PTX"
      VERBATIM)
    reception_embed("${gpu_gen}/reception_kernels_ptx.h" kReceptionPtx "${gpu_gen}/reception_kernels.ptx")
    list(APPEND gpu_sources "${gpu_gen}/reception_kernels_ptx.h")
    list(APPEND gpu_defs RECEPTION_HAS_CUDA)
    message(STATUS "Reception CUDA kernels: ${RECEPTION_NVCC}")
  else()
    message(STATUS "nvcc not found: CUDA projects will render Reception on the CPU")
  endif()
elseif(APPLE)
  enable_language(OBJCXX)
  reception_embed("${gpu_gen}/reception_kernels_metal.h" kReceptionMetalSource
               "${gpu_dir}/prelude_metal.metal" ${kernel_inputs})
  list(APPEND gpu_sources "${gpu_dir}/ReceptionGpuMetal.mm" "${gpu_gen}/reception_kernels_metal.h")
  set_source_files_properties("${gpu_dir}/ReceptionGpuMetal.mm" PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
else()
  return()
endif()

add_library(reception_gpu STATIC ${gpu_sources})
target_include_directories(reception_gpu PUBLIC "${gpu_dir}" "${gpu_gen}")
target_compile_definitions(reception_gpu PRIVATE ${gpu_defs})
target_link_libraries(reception_gpu PUBLIC reception_core)
if(WIN32)
  target_compile_definitions(reception_gpu PUBLIC NOMINMAX PRIVATE WIN32_LEAN_AND_MEAN)
  # Runs the GPU kernels on this machine (CUDA and OpenCL) and compares them with
  # the CPU renderer. Not part of ctest because CI has no GPU.
  add_executable(reception_gpu_check tools/gpu_check.cpp)
  target_link_libraries(reception_gpu_check PRIVATE reception_gpu)
else()
  target_link_libraries(reception_gpu PUBLIC "-framework Metal" "-framework Foundation")
endif()

# --- After Effects plugin --------------------------------------------------
if(NOT EXISTS "${AE_SDK_DIR}/Headers/AE_Effect.h")
  message(STATUS "AE SDK not found in ${AE_SDK_DIR}: skipping the Reception plugin")
  return()
endif()

set(pipl_src "${CMAKE_CURRENT_SOURCE_DIR}/plugin/ReceptionPiPL.r")
set(plugin_sources
  plugin/Reception.cpp
  "${AE_SDK_DIR}/Util/AEGP_SuiteHandler.cpp"
  "${AE_SDK_DIR}/Util/MissingSuiteError.cpp")

if(WIN32)
  set(pipl_rc "${CMAKE_CURRENT_BINARY_DIR}/ReceptionPiPL.rc")
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
    COMMENT "Compiling the Reception PiPL"
    VERBATIM)

  add_library(Reception MODULE ${plugin_sources} "${pipl_rc}")
  set_target_properties(Reception PROPERTIES SUFFIX ".aex" PREFIX "")
  target_compile_definitions(Reception PRIVATE MSWindows WIN32 _WINDOWS NOMINMAX)
elseif(APPLE)
  # The PiPL ships as a data-fork resource in Contents/Resources/Reception.rsrc.
  execute_process(COMMAND xcrun --show-sdk-path OUTPUT_VARIABLE macos_sdk
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(pipl_rsrc "${CMAKE_CURRENT_BINARY_DIR}/Reception.rsrc")
  add_custom_command(
    OUTPUT "${pipl_rsrc}"
    COMMAND xcrun Rez -o "${pipl_rsrc}" -define __MACH__ -useDF -script Roman
            -i "${AE_SDK_DIR}/Headers" -i "${AE_SDK_DIR}/Resources"
            -isysroot "${macos_sdk}" "${pipl_src}"
    DEPENDS "${pipl_src}"
    COMMENT "Compiling the Reception PiPL (Rez)"
    VERBATIM)
  set_source_files_properties("${pipl_rsrc}" PROPERTIES MACOSX_PACKAGE_LOCATION Resources)

  add_library(Reception MODULE ${plugin_sources} "${pipl_rsrc}")
  set_target_properties(Reception PROPERTIES
    BUNDLE TRUE
    BUNDLE_EXTENSION plugin
    OUTPUT_NAME Reception
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/plugin/Info.plist.in"
    MACOSX_BUNDLE_GUI_IDENTIFIER "com.xjfx.Reception"
    CXX_VISIBILITY_PRESET hidden)
endif()

target_include_directories(Reception PRIVATE
  "${AE_SDK_DIR}/Headers" "${AE_SDK_DIR}/Headers/SP"
  "${AE_SDK_DIR}/Resources" "${AE_SDK_DIR}/Util")
target_link_libraries(Reception PRIVATE reception_gpu)
