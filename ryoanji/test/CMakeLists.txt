include(sphexa_add_test)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(setup_GTest)

if (CMAKE_CUDA_COMPILER)
    set(testname ryoanji_unit_tests)
    add_executable(${testname}
            treebuilder.cu
            direct.cu
            multipole.cu
            warpscan.cu
            test_main.cpp)
    target_include_directories(${testname} PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(${testname} PUBLIC OpenMP::OpenMP_CXX cstone_tree GTest::gtest_main)
    set_source_files_properties(buildtree.cu PROPERTIES COMPILE_FLAGS -G)
    set_source_files_properties(direct.cu PROPERTIES COMPILE_FLAGS -G)
    set_source_files_properties(multipole.cu PROPERTIES COMPILE_FLAGS -G)
    # add_test(NAME RyoanjiGpuTests COMMAND ${testname})
    sphexa_add_test(${testname} EXECUTABLE ${testname} RANKS 1)
    install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/ryoanji/unit_tests)

    add_executable(ryoanji demo.cu)
    target_link_libraries(ryoanji PUBLIC cstone_tree)
    set_target_properties(ryoanji PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    # set_source_files_properties(demo.cu PROPERTIES COMPILE_FLAGS "-Xptxas -v")
endif ()

add_subdirectory(cpu)
