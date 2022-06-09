cmake_minimum_required(VERSION 3.17)

project(sphexa-cooling)
set(CMAKE_CXX_STANDARD 17)

#find_package(MPI)
#find_package(OpenMP REQUIRED)

#include(CTest)
#include(CheckLanguage)

if (BUILD_TESTING)
    add_subdirectory(test)
endif()
