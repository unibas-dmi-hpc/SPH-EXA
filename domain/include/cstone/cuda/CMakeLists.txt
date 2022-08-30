if(CMAKE_HIP_COMPILER)
    set_source_files_properties(gather.cu PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(gather_obj OBJECT gather.cu)
    target_include_directories(gather_obj PRIVATE ${PROJECT_SOURCE_DIR}/include)
endif()
