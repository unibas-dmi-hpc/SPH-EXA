cmake_minimum_required(VERSION 3.17)

project(sphexa-cooling)
set(CMAKE_CXX_STANDARD 17)

#find_package(MPI)
#find_package(OpenMP REQUIRED)

#include(CTest)
#include(CheckLanguage)
set(CSTONE_DIR ${PROJECT_SOURCE_DIR}/domain/include)
set(RYOANJI_DIR ${PROJECT_SOURCE_DIR}/ryoanji/src)
set(SPH_DIR ${PROJECT_SOURCE_DIR}/sph/include)

if (BUILD_TESTING)
    add_subdirectory(test)
endif()
