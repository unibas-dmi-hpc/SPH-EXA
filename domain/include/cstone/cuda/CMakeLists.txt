if(CMAKE_HIP_COMPILER)
    set_source_files_properties(gather.cu findneighbors.cu PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(gather_obj OBJECT gather.cu)

    add_library(cuda_find_neighbors_obj OBJECT findneighbors.cu)
    target_include_directories(cuda_find_neighbors_obj PRIVATE ${PROJECT_SOURCE_DIR}/include)
endif()
