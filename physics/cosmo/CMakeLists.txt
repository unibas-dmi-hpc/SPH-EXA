cmake_minimum_required(VERSION 3.17)

project(sphexa-cosmo)
set(CMAKE_CXX_STANDARD 20)

set(CSTONE_DIR ${CMAKE_SOURCE_DIR}/domain/include)

#include(CTest)
#include(CheckLanguage)
add_subdirectory(include/cosmo)

if (BUILD_TESTING)
    add_subdirectory(test)
endif()
