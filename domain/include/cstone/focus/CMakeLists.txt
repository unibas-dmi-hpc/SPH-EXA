if(CMAKE_HIP_COMPILER)
    set_source_files_properties(source_center_gpu.cu PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(source_center_gpu_obj OBJECT source_center_gpu.cu)
    target_include_directories(source_center_gpu_obj PRIVATE ${PROJECT_SOURCE_DIR}/include)
endif()