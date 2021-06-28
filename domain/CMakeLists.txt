cmake_minimum_required(VERSION 3.17)

project(cornerstone-octree CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(MPI)
find_package(CUDAToolkit)
find_package(OpenMP REQUIRED)

# if INTERFACE_LINK_OPTIONS for OpenMP_CXX is not set, we set
# INTERAFACE_LINK_OPTIONS = INTERFACE_COMPILE_OPTIONS
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

if(NOT CMAKE_CUDA_COMPILER)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_ARCHITECTURES OFF)
    endif()
endif()


if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_STANDARD 17)
    # use default nvcc cuda architectures
    set(CMAKE_CUDA_ARCHITECTURES OFF)

    if(OpenMP_FOUND)
        # workaround for openmp+cuda objects
        #set(CMAKE_CUDA_FLAGS "-Xcompiler ${OpenMP_CXX_FLAGS} ${CMAKE_CUDA_FLAGS}")
    endif()

else()
    message(STATUS "No CUDA support")
endif()

add_subdirectory(include)
add_subdirectory(test)
