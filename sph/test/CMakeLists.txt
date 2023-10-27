add_subdirectory(hydro_std)
add_subdirectory(hydro_turb)
add_subdirectory(hydro_ve)

set(testname sph_tests)
add_executable(${testname} sph_tests.cpp)
target_compile_options(${testname} PRIVATE -Wall -Wextra)

target_include_directories(${testname} PRIVATE ${CSTONE_DIR})
target_include_directories(${testname} PRIVATE ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(${testname} PRIVATE GTest::gtest_main)
add_test(NAME ${testname} COMMAND ${testname})