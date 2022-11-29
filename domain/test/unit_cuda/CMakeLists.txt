include(cstone_add_test)

if (CMAKE_HIP_COMPILER)
    set_source_files_properties(domain/domaindecomp_gpu.cu primitives/primitives_gpu.cu primitives/warpscan.cu
            primitives/clz.cu sfc/common.cu tree/btree.cu tree/csarray.cu tree/octree.cu PROPERTY LANGUAGE HIP)
endif ()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    set(testname component_units_cuda)
    add_executable(${testname}
            domain/domaindecomp_gpu.cu
            primitives/clz.cu
            primitives/primitives_gpu.cu
            primitives/warpscan.cu
            sfc/common.cu
            tree/btree.cu
            tree/csarray.cu
            tree/octree.cu
            test_main.cpp)
    target_include_directories(${testname} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_include_directories(${testname} PRIVATE ${PROJECT_SOURCE_DIR}/test)
    target_link_libraries(${testname} PUBLIC cstone_gpu OpenMP::OpenMP_CXX GTest::gtest_main)
    cstone_add_test(${testname} EXECUTABLE ${testname} RANKS 1)
    install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/unit_cuda)
    unset(testname)
endif()
