cmake_minimum_required(VERSION 3.14)

project(sphexa CXX)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS_RELEASE "-O2 -march=native -mtune=native -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O2 -DNDEBUG")

find_package(MPI)

add_subdirectory(test)
