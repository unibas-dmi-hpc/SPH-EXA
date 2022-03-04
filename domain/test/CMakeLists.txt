list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(setup_GTest)

add_subdirectory(coord_samples)
add_subdirectory(integration_mpi)
add_subdirectory(performance)
add_subdirectory(unit)
add_subdirectory(unit_cuda)
