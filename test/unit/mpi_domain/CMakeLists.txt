
add_executable(globaloctree globaloctree.cpp)
target_include_directories(globaloctree PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(globaloctree PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(globaloctree PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(globaloctree ${MPI_CXX_LIBRARIES} gtest_main)
add_test(NAME GlobalTreeTests COMMAND globaloctree)
