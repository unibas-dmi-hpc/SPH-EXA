
if (CMAKE_CUDA_COMPILER)
    add_library(ryoanji_upsweep upsweep.cu)
    target_include_directories(ryoanji_upsweep PUBLIC ${PROJECT_SOURCE_DIR}/src)
    target_include_directories(ryoanji_upsweep PRIVATE ${CSTONE_DIR})
endif ()