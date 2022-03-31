
set(testname ryoanji_cpu_unit_tests)
add_executable(${testname} cartesian_qpole.cpp traversal_cpu.cpp ../test_main.cpp)

target_compile_options(${testname} PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
target_include_directories(${testname} PRIVATE ${RYOANJI_TEST_INCLUDE_DIRS})
target_link_libraries(${testname} PRIVATE GTest::gtest_main OpenMP::OpenMP_CXX)
add_test(NAME ${testname} COMMAND ryoanji_cpu_unit_tests)
install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/ryoanji/cpu_unit_tests)

if (CMAKE_CUDA_COMPILER)
    set(testname ryoanji_unit_tests)
    add_executable(${testname}
            treebuilder.cu
            direct.cu
            multipole.cu
            warpscan.cu
            ../test_main.cpp)
    target_include_directories(${testname} PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(${testname} PUBLIC OpenMP::OpenMP_CXX cstone_tree GTest::gtest_main)
    set_source_files_properties(direct.cu PROPERTIES COMPILE_FLAGS -G)
    set_source_files_properties(multipole.cu PROPERTIES COMPILE_FLAGS -G)
    add_test(NAME ${testname} COMMAND ${testname})
    install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/ryoanji/unit_tests)
endif ()