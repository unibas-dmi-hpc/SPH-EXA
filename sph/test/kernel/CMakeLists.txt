include(sphexa_add_test)

set(UNIT_TESTS
        density_kern.cpp
        iad_kern.cpp
        momentum_energy.cpp
        test_main.cpp
        cooling_test.cpp
        )

set(testname kernel_tests)
add_executable(${testname} ${UNIT_TESTS})
target_compile_options(${testname} PRIVATE -Wall -Wextra)

target_include_directories(${testname} PRIVATE ${CSTONE_DIR} /usr/local/grackle/include/)
target_include_directories(${testname} PRIVATE ${PROJECT_SOURCE_DIR}/include)

target_compile_definitions(${testname} PUBLIC SPH_EXA_HAVE_H5PART)
target_include_directories(${testname} PUBLIC ${PROJECT_SOURCE_DIR}/extern/h5part)
target_link_libraries(${testname} PRIVATE H5Part ${HDF5_LIBRARIES})

set( CMAKE_BUILD_TYPE Debug )
target_link_libraries(${testname} PRIVATE GTest::gtest_main /usr/local/grackle/lib/libgrackle.a gfortran)


target_link_libraries(${testname} PRIVATE GTest::gtest_main)
sphexa_add_test(${testname} EXECUTABLE ${testname} RANKS 1)
#add_test(NAME KernelTests COMMAND ${testname})

install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/kernel)
