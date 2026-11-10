cmake_minimum_required(VERSION 3.20)

# macOS: universal binary (Apple Silicon + Intel). Must be set before project().
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
  set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "macOS architectures")
  set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS version")
endif()

project(RebornFX CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(MSVC)
  add_compile_options(/W4 /utf-8)
endif()

# Every plugin and tool lands in build/bin (build/bin/<Config> with multi-config generators).
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

enable_testing()

# The Adobe After Effects SDK is not redistributable, so it is not in the
# repository: pass -DAE_SDK_DIR=<sdk>/Examples, set the AE_SDK_DIR environment
# variable, or copy Headers/, Util/ and Resources/ into third_party/ae_sdk
# (ignored by git). Without it, only the cores and their tests are built.
if(DEFINED ENV{AE_SDK_DIR})
  set(ae_sdk_default "$ENV{AE_SDK_DIR}")
else()
  set(ae_sdk_default "${CMAKE_SOURCE_DIR}/third_party/ae_sdk")
endif()
set(AE_SDK_DIR "${ae_sdk_default}"
    CACHE PATH "Folder with the After Effects SDK Headers/, Util/ and Resources/")

add_subdirectory(plugins/tremor)
