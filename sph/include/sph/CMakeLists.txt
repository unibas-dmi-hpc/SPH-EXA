add_subdirectory(hydro_std)
add_subdirectory(hydro_turb)
add_subdirectory(hydro_ve)

if (CMAKE_HIP_COMPILER)
    set_source_files_properties(positions_gpu.cu update_h_gpu.cu PROPERTIES LANGUAGE HIP)
endif()

if (CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(sph_gpu
            positions_gpu.cu
            update_h_gpu.cu
            $<TARGET_OBJECTS:hydro_std_gpu>
            $<TARGET_OBJECTS:hydro_turb_gpu>
            $<TARGET_OBJECTS:hydro_ve_gpu>)
    target_include_directories(sph_gpu PUBLIC ${CSTONE_DIR} ${PROJECT_SOURCE_DIR}/include)
    set_target_properties(sph_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif ()

