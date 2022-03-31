
if (CMAKE_CUDA_COMPILER)
    add_library(cstone_tree treebuilder.cu)
    target_include_directories(cstone_tree PUBLIC ${PROJECT_SOURCE_DIR}/src)
    target_include_directories(cstone_tree PUBLIC ${CSTONE_DIR})
endif ()
