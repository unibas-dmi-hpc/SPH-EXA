if(CMAKE_HIP_COMPILER)
    set_source_files_properties(induction_dissipation_gpu.cu PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(magneto_ve_gpu OBJECT
            iad_div_curl.cu
            magnetic_momentum_energy_gpu.cu
            induction_dissipation_gpu.cu
            time_integration_gpu.cu)
    target_include_directories(magneto_ve_gpu PRIVATE ${CSTONE_DIR} ${PROJECT_SOURCE_DIR}/include ${COOLING_DIR})
    target_compile_definitions(magneto_ve_gpu PRIVATE USE_CUDA)
endif()

