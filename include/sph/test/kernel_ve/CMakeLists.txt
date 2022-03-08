set(UNIT_TESTS
        rho_zero_kern.cpp
        density_kern.cpp
        iad_kern.cpp
        divv_curlv_kern.cpp
        av_switches_kern.cpp
        momentum_energy_kern.cpp
        test_main.cpp
        )

set(kernel_target kernel_tests_ve)
add_executable(${kernel_target} ${UNIT_TESTS})
target_compile_options(${kernel_target} PRIVATE -Wall -Wextra)

#target_include_directories(${kernel_target} PRIVATE ../../../)
target_include_directories(${kernel_target} PRIVATE ${PROJECT_SOURCE_DIR}/domain/include)
target_include_directories(${kernel_target} PRIVATE ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(${kernel_target} PRIVATE gtest_main)
add_test(NAME KernelTestsVE COMMAND ${kernel_target})

install(TARGETS ${kernel_target} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/kernel)
