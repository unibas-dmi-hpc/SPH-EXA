cmake_minimum_required(VERSION 3.14)

<<<<<<< HEAD
<<<<<<< HEAD
project(sphexa CXX)
set(CMAKE_CXX_STANDARD 17)
=======
project(sphexa CXX CUDA)
=======
project(sphexa CXX)
>>>>>>> link tests against MPI
set(CMAKE_CXX_STANDARD 14)
>>>>>>> reenabled unit tests (#44)

set(CMAKE_CXX_FLAGS_RELEASE "-O2 -march=native -mtune=native -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O2 -DNDEBUG")

<<<<<<< HEAD
<<<<<<< HEAD
find_package(MPI)
find_package(CUDAToolkit)
find_package(OpenMP REQUIRED)

include(CTest)

add_subdirectory(domain)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_subdirectory(include)
else()
    message(STATUS "No CUDA support")
endif()

add_subdirectory(src)
=======
=======
find_package(MPI)
include(CTest)

>>>>>>> link tests against MPI
add_subdirectory(test)
>>>>>>> reenabled unit tests (#44)
