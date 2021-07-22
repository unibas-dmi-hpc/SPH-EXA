
add_executable(octree_perf octree.cpp)
target_include_directories(octree_perf PRIVATE ../../include)
target_include_directories(octree_perf PRIVATE ../)
target_link_libraries(octree_perf PRIVATE OpenMP::OpenMP_CXX)

add_executable(peers_perf peers.cpp)
target_include_directories(peers_perf PRIVATE ../../include)
target_include_directories(peers_perf PRIVATE ../)
target_link_libraries(peers_perf PRIVATE OpenMP::OpenMP_CXX)

add_executable(scan_perf scan.cpp)
target_include_directories(scan_perf PRIVATE ../../include)
target_include_directories(scan_perf PRIVATE ../)
target_link_libraries(scan_perf PRIVATE OpenMP::OpenMP_CXX)

add_executable(hilbert_perf hilbert.cpp)
target_include_directories(hilbert_perf PRIVATE ../../include)
target_include_directories(hilbert_perf PRIVATE ../)
target_link_libraries(hilbert_perf PRIVATE OpenMP::OpenMP_CXX)

if(CMAKE_CUDA_COMPILER)
    add_executable(cudaNeighborsTest $<TARGET_OBJECTS:cuda_find_neighbors_obj> neighbor_driver.cu)
    target_include_directories(cudaNeighborsTest PRIVATE ../../include)
    target_include_directories(cudaNeighborsTest PRIVATE ../)

    target_link_libraries(cudaNeighborsTest PRIVATE ${CUDA_RUNTIME_LIBRARY} OpenMP::OpenMP_CXX)

    add_executable(octree_perf_gpu octree.cu)
    target_include_directories(octree_perf_gpu PRIVATE ../../include)
    target_include_directories(octree_perf_gpu PRIVATE ../)

    add_executable(binary_perf_gpu btreetraversal.cu)
    target_include_directories(binary_perf_gpu PRIVATE ../../include)
    target_include_directories(binary_perf_gpu PRIVATE ../)
    target_link_libraries(binary_perf_gpu PRIVATE OpenMP::OpenMP_CXX)

    add_executable(hilbert_perf_gpu hilbert.cu)
    target_include_directories(hilbert_perf_gpu PRIVATE ../../include)
    target_include_directories(hilbert_perf_gpu PRIVATE ../)
endif()
