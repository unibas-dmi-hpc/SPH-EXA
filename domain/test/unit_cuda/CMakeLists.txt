
if(CMAKE_CUDA_COMPILER)

    add_executable(component_units_cuda
            $<TARGET_OBJECTS:gather_obj>
            btree.cu
            octree.cu
            octree_internal.cu
            gather.cpp
            test_main.cpp)
    target_include_directories(component_units_cuda PRIVATE ../../include)
    target_include_directories(component_units_cuda PRIVATE ../)
    target_link_libraries(component_units_cuda PUBLIC OpenMP::OpenMP_CXX gtest_main)
    install(TARGETS component_units_cuda RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/unit_cuda)
endif()
