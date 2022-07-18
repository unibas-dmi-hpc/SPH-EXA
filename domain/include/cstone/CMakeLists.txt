add_subdirectory(cuda)
add_subdirectory(halos)

if (CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(cstone_gpu $<TARGET_OBJECTS:gather_send_obj>
                           $<TARGET_OBJECTS:gather_obj>
                           $<TARGET_OBJECTS:cuda_find_neighbors_obj>)
endif ()