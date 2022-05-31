
set(UNIT_TESTS
        init/grid.cpp
        init/isobaric_cube.cpp
        io/arg_parser.cpp
        observables/gravitational_waves.cpp
        sphexa/particles_data.cpp
        test_main.cpp)

set(exename frontend_units)
add_executable(${exename} ${UNIT_TESTS})
target_compile_options(${exename} PRIVATE -Wall -Wextra -Wno-unknown-pragmas)

target_include_directories(${exename} PRIVATE ${SPH_DIR} ${CSTONE_DIR} ${PROJECT_SOURCE_DIR}/main/src)
target_link_libraries(${exename} PRIVATE GTest::gtest_main)
add_test(NAME FrontendUnits COMMAND ${exename})

if(CMAKE_CUDA_COMPILER)
    set(testname frontend_units_cuda)
    add_executable(${testname}
            cuda/gpu_particles_data.cpp
            test_main.cpp)
    target_include_directories(${testname} PRIVATE ${SPH_DIR} ${CSTONE_DIR} ${PROJECT_SOURCE_DIR}/main/src)
    target_link_libraries(${testname} PUBLIC sph_gpu OpenMP::OpenMP_CXX CUDA::cudart GTest::gtest_main)
    add_test(NAME FrontendUnitsCuda COMMAND ${exename})
    unset(testname)
endif()
