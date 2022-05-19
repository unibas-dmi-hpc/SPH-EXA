set(UNIT_TESTS
        xmass_kern.cpp
        ve_norm_gradh.cpp
        iad_kern.cpp
        divv_curlv_kern.cpp
        av_switches_kern.cpp
        momentum_energy_kern.cpp
        test_main.cpp
        )

set(testname kernel_tests_ve)
add_executable(${testname} ${UNIT_TESTS})
target_compile_options(${testname} PRIVATE -Wall -Wextra)

target_include_directories(${testname} PRIVATE ${CSTONE_DIR})
target_include_directories(${testname} PRIVATE ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(${testname} PRIVATE GTest::gtest_main)
add_test(NAME ${testname} COMMAND ${testname})

install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/hydro)
