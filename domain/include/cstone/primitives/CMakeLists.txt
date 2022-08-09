if(CMAKE_HIP_COMPILER)
    set_source_files_properties(gather_gpu.cu PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(gather_gpu_obj OBJECT gather_gpu.cu)
    target_include_directories(gather_gpu_obj PRIVATE ${PROJECT_SOURCE_DIR}/include)
endif()