list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(setup_GTest)
include(ryoanji_add_test)

set(RYOANJI_TEST_INCLUDE_DIRS ${CSTONE_DIR} ${CSTONE_TEST_DIR} ${PROJECT_SOURCE_DIR}/src)

if (CMAKE_CUDA_COMPILER)
    add_executable(ryoanji_demo demo.cu)
    target_link_libraries(ryoanji_demo PUBLIC cstone_tree)
    set_target_properties(ryoanji_demo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    # set_source_files_properties(demo.cu PROPERTIES COMPILE_FLAGS "-Xptxas -v")
    install(TARGETS ryoanji_demo RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/ryoanji/ryoanji_demo)
endif ()

# CPU unit test suite
set(testname ryoanji_cpu_unit_tests)
add_executable(${testname}
        nbody/cartesian_qpole.cpp
        nbody/traversal_cpu.cpp
        test_main.cpp)

target_compile_options(${testname} PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
target_include_directories(${testname} PRIVATE ${RYOANJI_TEST_INCLUDE_DIRS})
target_link_libraries(${testname} PRIVATE GTest::gtest_main OpenMP::OpenMP_CXX)
add_test(NAME ${testname} COMMAND ryoanji_cpu_unit_tests)
install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/ryoanji/cpu_unit_tests)

# GPU unit test suite
if (CMAKE_CUDA_COMPILER)
    set(testname ryoanji_unit_tests)
    add_executable(${testname}
            nbody/direct.cu
            nbody/multipole.cu
            nbody/warpscan.cu
            test_main.cpp)
    target_include_directories(${testname} PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(${testname} PUBLIC OpenMP::OpenMP_CXX cstone_tree GTest::gtest_main)
    set_source_files_properties(nbody/direct.cu PROPERTIES COMPILE_FLAGS -G)
    set_source_files_properties(nbody/multipole.cu PROPERTIES COMPILE_FLAGS -G)
    add_test(NAME ${testname} COMMAND ${testname})
    install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/ryoanji/unit_tests)
endif ()

add_subdirectory(interface)