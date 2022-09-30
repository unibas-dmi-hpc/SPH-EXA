if(CMAKE_HIP_COMPILER)
    set_source_files_properties(reallocate.cu PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(reallocate_obj OBJECT reallocate.cu)
    target_include_directories(reallocate_obj PRIVATE ${PROJECT_SOURCE_DIR}/include)
endif()