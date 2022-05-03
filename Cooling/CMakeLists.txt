#cmake_minimum_required(VERSION 3.17)

#project(sphexa-cooling)
#set(CMAKE_CXX_STANDARD 17)

#find_package(MPI)
#find_package(OpenMP REQUIRED)


#include(CTest)
#include(CheckLanguage)


#include(sph)
set(CSTONE_DIR ${CMAKE_SOURCE_DIR}/domain/include)
add_subdirectory(include)
#if (BUILD_TESTING)
    add_subdirectory(test)
#endif()
