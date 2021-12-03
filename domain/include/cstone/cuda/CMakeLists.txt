
if(CMAKE_CUDA_COMPILER)
    add_library(cuda_find_neighbors_obj OBJECT findneighbors.cu)
    target_include_directories(cuda_find_neighbors_obj PRIVATE ${PROJECT_SOURCE_DIR}/include)

    add_library(gather_obj OBJECT gather.cu)

    add_library(thrust_sort_scan_obj OBJECT thrust_sort_scan.cu)
    target_include_directories(thrust_sort_scan_obj PRIVATE ${PROJECT_SOURCE_DIR}/include)
endif()
