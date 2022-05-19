if(CMAKE_HIP_COMPILER)
    set_source_files_properties(density_gpu.cu iad_gpu.cu momentum_energy_gpu.cu PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(hydro_std_gpu OBJECT density_gpu.cu iad_gpu.cu momentum_energy_gpu.cu)
    target_include_directories(hydro_std_gpu PRIVATE ${CSTONE_DIR} ${PROJECT_SOURCE_DIR}/include)
    target_compile_definitions(hydro_std_gpu PRIVATE USE_CUDA)
    # set_property(TARGET hydro_std_gpu PROPERTY HIP_ARCHITECTURES gfx908)
endif()

