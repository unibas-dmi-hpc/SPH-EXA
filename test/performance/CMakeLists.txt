
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    add_executable(cudaNeighborsTest $<TARGET_OBJECTS:cudaFindNeighbors> neighbor_driver.cpp)
    target_include_directories(cudaNeighborsTest PRIVATE ../../include)
    target_include_directories(cudaNeighborsTest PRIVATE ../)

    target_link_libraries(cudaNeighborsTest PUBLIC CUDA::cudart OpenMP::OpenMP_CXX)

    add_executable(gather_test $<TARGET_OBJECTS:gatherObj> gather_driver.cpp)
    target_include_directories(gather_test PRIVATE ../../include)
    target_include_directories(gather_test PRIVATE ../)

    target_link_libraries(gather_test PUBLIC CUDA::cudart OpenMP::OpenMP_CXX)

else()
    message(STATUS "No CUDA support")
endif()

add_executable(octree_perf octree.cpp)

target_include_directories(octree_perf PRIVATE ../../include)
target_include_directories(octree_perf PRIVATE ../)
target_link_libraries(octree_perf PUBLIC OpenMP::OpenMP_CXX)