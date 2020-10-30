
add_executable(coordinate_test random.cpp test_main.cpp)
target_include_directories(coordinate_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(coordinate_test gtest_main)
add_test(NAME CoordinateTests COMMAND coordinate_test)
