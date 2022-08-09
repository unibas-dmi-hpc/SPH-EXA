if(CMAKE_HIP_COMPILER)
    set_source_files_properties(sfc.cu PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(sfc_obj OBJECT sfc.cu)
    target_include_directories(sfc_obj PRIVATE ${PROJECT_SOURCE_DIR}/include)
endif()