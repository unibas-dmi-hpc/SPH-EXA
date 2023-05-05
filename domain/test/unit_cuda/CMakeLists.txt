include(cstone_add_test)

if(CMAKE_HIP_COMPILER)
    set_source_files_properties(domain/domaindecomp_gpu.cu primitives/primitives_gpu.cu primitives/warpscan.cu
            primitives/clz.cu sfc/common.cu traversal/groups.cu tree/btree.cu tree/csarray.cu tree/octree.cu
            PROPERTIES LANGUAGE HIP)
endif()

set(testname component_units_cuda)

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(gpu_unit_obj OBJECT
            domain/domaindecomp_gpu.cu
            primitives/clz.cu
            primitives/primitives_gpu.cu
            primitives/warpscan.cu
            sfc/common.cu
            traversal/groups.cu
            tree/btree.cu
            tree/csarray.cu
            tree/octree.cu)
    target_link_libraries(gpu_unit_obj PUBLIC cstone_gpu OpenMP::OpenMP_CXX GTest::gtest_main)
    target_include_directories(gpu_unit_obj PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_include_directories(gpu_unit_obj PRIVATE ${PROJECT_SOURCE_DIR}/test)

    add_executable(${testname} test_main.cpp)
    target_link_libraries(${testname} PUBLIC gpu_unit_obj)
    cstone_add_test(${testname} EXECUTABLE ${testname} RANKS 1)
    install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/unit_cuda)
endif()

if(CMAKE_HIP_COMPILER)
    target_link_libraries(gpu_unit_obj PUBLIC hip::host)
    target_compile_definitions(gpu_unit_obj PRIVATE USE_CUDA THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)
    set_target_properties(${testname} PROPERTIES LINKER_LANGUAGE CXX)
endif()

unset(testname)
