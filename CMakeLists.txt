cmake_minimum_required(VERSION 3.14)

project(cornerstone-octree CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(MPI)
find_package(CUDAToolkit)
find_package(OpenMP REQUIRED)

include(CTest)
include(CheckLanguage)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_subdirectory(include)
else()
    message(STATUS "No CUDA support")
endif()

add_subdirectory(test)
