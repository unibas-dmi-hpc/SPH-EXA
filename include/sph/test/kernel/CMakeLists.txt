set(UNIT_TESTS
        density.cpp
        iad.cpp
        momentum_energy.cpp
        test_main.cpp
        )

set(kernel_target kernel_tests)
add_executable(${kernel_target} ${UNIT_TESTS})
target_compile_options(${kernel_target} PRIVATE -Wall -Wextra)

#target_include_directories(${kernel_target} PRIVATE ../../../)
target_include_directories(${kernel_target} PRIVATE ${PROJECT_SOURCE_DIR}/domain/include)
target_include_directories(${kernel_target} PRIVATE ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(${kernel_target} PRIVATE gtest_main)
add_test(NAME KernelTests COMMAND ${kernel_target})

install(TARGETS ${kernel_target} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/kernel)
