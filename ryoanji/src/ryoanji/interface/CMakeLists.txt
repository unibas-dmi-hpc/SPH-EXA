
if (CMAKE_CUDA_COMPILER)
    add_library(cstone_tree treebuilder.cu)
    target_include_directories(cstone_tree PUBLIC ${PROJECT_SOURCE_DIR}/src)
    target_include_directories(cstone_tree PUBLIC ${CSTONE_DIR})

    add_library(ryoanji multipole_holder.cu)
    target_include_directories(ryoanji PUBLIC ${PROJECT_SOURCE_DIR}/src ${CSTONE_DIR} ${MPI_CXX_INCLUDE_PATH})
    target_link_libraries(ryoanji PUBLIC ${MPI_CXX_LIBRARIES})
endif ()
