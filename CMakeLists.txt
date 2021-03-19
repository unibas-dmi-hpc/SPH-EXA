cmake_minimum_required(VERSION 3.17)

project(cornerstone-octree CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(MPI)
find_package(CUDAToolkit)
find_package(OpenMP REQUIRED)

include(CTest)

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(CUDAToolkit_FOUND)
    enable_language(CUDA)

    # does nothing
    set(CMAKE_CUDA_STANDARD 17)
    # workaround for CMAKE_CUDA_STANDARD not setting the required flag
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-std=c++17>)

    add_subdirectory(include)
else()
    message(STATUS "No CUDA support")
endif()

add_subdirectory(test)
