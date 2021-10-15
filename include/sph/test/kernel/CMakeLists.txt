
set(UNIT_TESTS
        density.cpp
        test_main.cpp
        )

set(kernel_target kernel_tests)
add_executable(${kernel_target} ${UNIT_TESTS})
target_compile_options(${kernel_target} PRIVATE -Wall -Wextra)

target_include_directories(${kernel_target} PRIVATE ../../../)
target_link_libraries(${kernel_target} PRIVATE gtest_main)
add_test(NAME KernelTests COMMAND ${kernel_target})
