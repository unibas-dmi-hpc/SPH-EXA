cmake_minimum_required(VERSION 3.17)

project(sphexa-cooling)
set(CMAKE_CXX_STANDARD 20)

set(CSTONE_DIR ${CMAKE_SOURCE_DIR}/domain/include)

#include(CTest)
#include(CheckLanguage)

if (BUILD_TESTING)
    add_subdirectory(test)
endif()
