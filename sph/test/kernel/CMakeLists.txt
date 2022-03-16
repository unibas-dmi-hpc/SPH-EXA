include(sphexa_add_test)

set(UNIT_TESTS
        density_kern.cpp
        iad_kern.cpp
        momentum_energy.cpp
        cooling_test.cpp
        test_main.cpp
)

set(GRACKLE_DIR ${PROJECT_SOURCE_DIR}/../extern/grackle/src/clib)
set(testname kernel_tests)
add_executable(${testname} ${UNIT_TESTS})
target_compile_options(${testname} PRIVATE -Wall -Wextra)
target_compile_definitions(${testname} PUBLIC SPH_EXA_HAVE_H5PART)

target_include_directories(${testname} PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(${testname} PRIVATE ${PROJECT_SOURCE_DIR}/include/sph)
target_include_directories(${testname} PUBLIC ${PROJECT_SOURCE_DIR}/extern/h5part)
target_include_directories(${testname} PRIVATE ${CSTONE_DIR})
target_include_directories(${testname} PRIVATE ${GRACKLE_DIR})

target_link_libraries(${testname} PRIVATE H5Part ${HDF5_LIBRARIES})
target_link_libraries(${testname} PRIVATE grackle)
target_link_libraries(${testname} PRIVATE GTest::gtest_main)

sphexa_add_test(${testname} EXECUTABLE ${testname} RANKS 1)

install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/kernel)
