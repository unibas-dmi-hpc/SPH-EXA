
set(UNIT_TESTS
        init/grid.cpp
        init/isobaric_cube.cpp
        io/arg_parser.cpp
        observables/gravitational_waves.cpp
        test_main.cpp)

set(exename frontend_units)
add_executable(${exename} ${UNIT_TESTS})
target_compile_options(${exename} PRIVATE -Wall -Wextra -Wno-unknown-pragmas)

target_include_directories(${exename} PRIVATE ${CSTONE_DIR} ${PROJECT_SOURCE_DIR}/main/src)
target_link_libraries(${exename} PRIVATE GTest::gtest_main)
add_test(NAME FrontendUnits COMMAND ${exename})
