cmake_minimum_required(VERSION 3.14)

project(sphexa CXX)
project(cornerstone-octree CXX)

set(CMAKE_CXX_FLAGS_RELEASE "-O2 -march=native -mtune=native -DNDEBUG")

find_package(MPI)
include(CTest)

add_subdirectory(test)
