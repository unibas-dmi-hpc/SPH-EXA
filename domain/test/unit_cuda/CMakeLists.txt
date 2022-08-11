include(cstone_add_test)

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    set(testname component_units_cuda)
    add_executable(${testname}
            $<TARGET_OBJECTS:gather_obj>
            btree.cu
            octree.cu
            octree_internal.cu
            gather.cpp
            test_main.cpp)
    target_include_directories(${testname} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_include_directories(${testname} PRIVATE ${PROJECT_SOURCE_DIR}/test)
    target_link_libraries(${testname} PUBLIC OpenMP::OpenMP_CXX GTest::gtest_main)
    cstone_add_test(${testname} EXECUTABLE ${testname} RANKS 1)
    install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/unit_cuda)
    unset(testname)
endif()
