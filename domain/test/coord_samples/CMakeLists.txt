
add_executable(coordinate_test random.cpp test_main.cpp)
target_include_directories(coordinate_test PRIVATE ../../include)
target_link_libraries(coordinate_test GTest::gtest_main)
target_compile_options(coordinate_test PRIVATE -Wno-unknown-pragmas)
add_test(NAME CoordinateTests COMMAND coordinate_test)

install(TARGETS coordinate_test RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/coord_samples)
