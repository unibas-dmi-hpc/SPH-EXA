cmake_minimum_required(VERSION 3.17)

project(sphexa CXX C)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fno-math-errno -march=native -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

option(BUILD_TESTING "build unit and integration tests" ON)
option(SPH_EXA_WITH_HIP "Build SPH-EXA with HIP" OFF)

if (SPH_EXA_WITH_HIP)
    cmake_minimum_required(VERSION 3.21)
endif()


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

if(SPH_EXA_WITH_HIP)
# {{{ HIP
if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
# set(HIP_CLANG_NUM_PARALLEL_JOBS 2)
find_package(HIP QUIET)
if(HIP_FOUND)
    message(STATUS "Found HIP: " ${HIP_VERSION})
    check_language(HIP)
    # find_package(HIP)
    if(CMAKE_HIP_COMPILER)
        enable_language(HIP)
        set(CMAKE_HIP_FLAGS "-fPIC -fgpu-rdc -w")
    endif()
endif()
# }}}
endif()

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES OFF)
else()
    message(STATUS "No CUDA support")
endif()

# {{{ i/o
option(SPH_EXA_WITH_H5PART "Enable HDF5 IO using H5Part library" OFF)
if (SPH_EXA_WITH_H5PART)
    find_package(HDF5 REQUIRED)
    add_subdirectory(./extern/h5part)
endif()

set(INSITU None CACHE STRING "Enable in-situ support")
set_property(CACHE INSITU PROPERTY STRINGS None Catalyst Ascent)

if(INSITU STREQUAL "Catalyst")
	find_package(catalyst REQUIRED PATHS "/scratch/snx3000/jfavre/daint/software/ParaView/5.10.0-CrayGNU-20.11-RC2-EGL-python3/lib/cmake/paraview-5.10")
elseif(INSITU STREQUAL "Ascent")
    find_package(Ascent REQUIRED PATHS "/apps/daint/UES/Ascent/ascent-install/lib/cmake/ascent")
endif()
# }}}

add_subdirectory(include)
add_subdirectory(domain)

if (CMAKE_CUDA_COMPILER AND BUILD_RYOANJI)
    add_subdirectory(ryoanji)
endif()

add_subdirectory(src)
if (BUILD_TESTING)
    add_subdirectory(test)
endif()
