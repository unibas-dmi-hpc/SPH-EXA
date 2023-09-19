include(cstone_add_performance_test)

cstone_add_performance_test(octree.cpp octree_perf)
cstone_add_performance_test(peers.cpp peers_perf)
cstone_add_performance_test(scan.cpp scan_perf)

# only scan.cpp provides some coverage beyond the unit tests
cstone_add_test(scan_perf EXECUTABLE scan_perf RANKS 1)

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    # uses the OpenMP implementation of thrust::sort_by_key
    cstone_add_performance_test(hilbert.cpp hilbert_perf)
    target_compile_definitions(hilbert_perf PRIVATE THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP)
    if (CMAKE_CUDA_COMPILER)
        target_link_libraries(hilbert_perf PRIVATE CUDA::cudart)
    else ()
        target_compile_definitions(hilbert_perf PRIVATE THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)
        target_link_libraries(hilbert_perf PRIVATE hip::host)
    endif ()
endif ()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    cstone_add_cuda_performance_test(neighbor_driver.cu neighbors_test_gpu)
    cstone_add_cuda_performance_test(octree.cu octree_perf_gpu)
    cstone_add_cuda_performance_test(hilbert.cu hilbert_perf_gpu)

    if (CMAKE_CUDA_COMPILER)
        set_source_files_properties(octree.cu PROPERTIES COMPILE_OPTIONS --expt-relaxed-constexpr)
    endif()
endif()
