
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_executable(cudaNeighborsTest findneighbors.cu neighbor_driver.cpp)
    target_include_directories(cudaNeighborsTest PRIVATE ../../include)
    target_include_directories(cudaNeighborsTest PRIVATE ../)

    target_link_libraries(cudaNeighborsTest PUBLIC CUDA::cudart)
    target_link_libraries(cudaNeighborsTest PUBLIC OpenMP::OpenMP_CXX)

else()
    message(STATUS "No CUDA support")
endif()

add_executable(octree_perf octree.cpp)

target_include_directories(octree_perf PRIVATE ../../include)
target_include_directories(octree_perf PRIVATE ../)
target_link_libraries(octree_perf PUBLIC OpenMP::OpenMP_CXX)