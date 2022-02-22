include(sphexa_add_test)

set(UNIT_TESTS
        density.cpp
        iad.cpp
        momentum_energy.cpp
        test_main.cpp
        )

set(testname kernel_tests)
add_executable(${testname} ${UNIT_TESTS})
target_compile_options(${testname} PRIVATE -Wall -Wextra)

target_include_directories(${testname} PRIVATE ${PROJECT_SOURCE_DIR}/domain/include)
target_include_directories(${testname} PRIVATE ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(${testname} PRIVATE GTest::gtest_main)
sphexa_add_test(${testname} EXECUTABLE ${testname} RANKS 1)
#add_test(NAME KernelTests COMMAND ${testname})

install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR})
