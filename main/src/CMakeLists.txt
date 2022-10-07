
set(SPH_EXA_INCLUDE_DIRS ${CSTONE_DIR} ${COOLING_DIR} ${RYOANJI_DIR} ${SPH_DIR}
    ${PROJECT_SOURCE_DIR}/main/src
    ${CMAKE_BINARY_DIR}/main/src
    ${MPI_CXX_INCLUDE_PATH})

add_subdirectory(observables)
add_subdirectory(sphexa)

if (BUILD_ANALYTICAL)
    add_subdirectory(analytical_solutions)
endif ()

include(setup_gitinfo)
configure_file(
  version.h.in
  ${CMAKE_BINARY_DIR}/main/src/version.h
)
