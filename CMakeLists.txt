cmake_minimum_required(VERSION 3.17)

project(cornerstone-octree CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(MPI)
find_package(CUDAToolkit)
find_package(OpenMP REQUIRED)

include(CTest)
include(CheckLanguage)

if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    add_subdirectory(include)
else()
    message(STATUS "No CUDA support")
endif()

add_subdirectory(test)
