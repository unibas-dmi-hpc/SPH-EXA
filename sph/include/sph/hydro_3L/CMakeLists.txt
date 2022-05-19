if(CMAKE_HIP_COMPILER)
    set_source_files_properties(cuda_density.cu cuda_iad.cu cuda_grad_p.cu PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(hydro_3L_gpu OBJECT cuda_density.cu cuda_iad.cu cuda_grad_p.cu)
    target_include_directories(hydro_3L_gpu PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_include_directories(hydro_3L_gpu PUBLIC ${CSTONE_DIR})
    target_compile_definitions(hydro_3L_gpu PRIVATE USE_CUDA)
    # set_property(TARGET hydro_3L_gpu PROPERTY HIP_ARCHITECTURES gfx908)
endif()

