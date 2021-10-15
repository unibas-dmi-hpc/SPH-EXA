cmake_minimum_required(VERSION 3.17)

project(sphexa CXX)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_RELEASE "-O2 -march=native -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O2 -DNDEBUG")

find_package(MPI)
find_package(OpenMP REQUIRED)

# if INTERFACE_LINK_OPTIONS for OpenMP_CXX is not set, we set
# INTERFACE_LINK_OPTIONS = INTERFACE_COMPILE_OPTIONS
# i.e. we add e.g. the -fopenmp flag to the linker
get_property(OpenMP_CXX_INTERFACE_LINK_OPTIONS TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_LINK_OPTIONS)
if (NOT OpenMP_CXX_INTERFACE_LINK_OPTIONS)
    # fill in using INTERFACE_COMPILE_OPTIONS
    get_property(OpenMP_CXX_INTERFACE_LINK_OPTIONS TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_COMPILE_OPTIONS)
    set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_LINK_OPTIONS ${OpenMP_CXX_INTERFACE_LINK_OPTIONS})
endif()

include(CTest)
include(CheckLanguage)

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES OFF)
else()
    message(STATUS "No CUDA support")
endif()

add_subdirectory(include)
add_subdirectory(domain)

if (CMAKE_CUDA_COMPILER)
    add_subdirectory(ryoanji)
endif()

add_subdirectory(src)
add_subdirectory(test)
