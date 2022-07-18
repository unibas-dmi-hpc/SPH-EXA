if(CMAKE_HIP_COMPILER)
    set_source_files_properties(gather_send.cu PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(gather_send_obj OBJECT gather_send.cu)
    target_include_directories(gather_send_obj PRIVATE ${PROJECT_SOURCE_DIR}/include)
endif()