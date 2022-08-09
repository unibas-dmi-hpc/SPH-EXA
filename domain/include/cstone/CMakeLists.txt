add_subdirectory(cuda)
add_subdirectory(halos)
add_subdirectory(primitives)
add_subdirectory(sfc)
add_subdirectory(util)

if (CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(cstone_gpu $<TARGET_OBJECTS:gather_scatter_obj>
                           $<TARGET_OBJECTS:gather_obj>
                           $<TARGET_OBJECTS:primitives_gpu_obj>
                           $<TARGET_OBJECTS:sfc_obj>
                           $<TARGET_OBJECTS:thrust_alloc_obj>
                           $<TARGET_OBJECTS:cuda_find_neighbors_obj>)
endif ()