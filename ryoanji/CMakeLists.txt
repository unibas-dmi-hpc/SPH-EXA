cmake_minimum_required(VERSION 3.17)

project(ryoanji CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "35;60")

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

add_executable(ryoanji_cs thrust.cu serial_cs.cu)
target_include_directories(ryoanji_cs PRIVATE include)
target_include_directories(ryoanji_cs PRIVATE ../domain/include)
set_target_properties(ryoanji_cs PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_source_files_properties(serial_cs.cu PROPERTIES COMPILE_FLAGS "-Xptxas -v")

add_subdirectory(test)
