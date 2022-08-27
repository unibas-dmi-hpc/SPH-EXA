if(CMAKE_HIP_COMPILER)
    set_source_files_properties(thrust_alloc.cu PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(thrust_alloc_obj OBJECT thrust_alloc.cu)
    target_include_directories(thrust_alloc_obj PRIVATE ${PROJECT_SOURCE_DIR}/include)
endif()