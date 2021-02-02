
if(CMAKE_CUDA_COMPILER)
    add_library(cuda_sph OBJECT cudaFindNeighbors.cu cudaDensity.cu cudaIAD.cu cudaMomentumAndEnergyIAD.cu)
    target_include_directories(cuda_sph PRIVATE ${PROJECT_SOURCE_DIR}/include)

    target_compile_definitions(cuda_sph PRIVATE USE_CUDA)
    target_compile_options(cuda_sph PRIVATE --expt-relaxed-constexpr)
    target_compile_features(cuda_sph PRIVATE cxx_std_14)
    set_target_properties(cuda_sph PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
