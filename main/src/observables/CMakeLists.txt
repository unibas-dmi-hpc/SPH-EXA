if(CMAKE_HIP_COMPILER)
    set_source_files_properties(conserved_gpu.cu PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(observables_gpu conserved_gpu.cu)
    target_include_directories(observables_gpu PRIVATE ${CSTONE_DIR})
endif()