
set(CSTONE_DIR ${PROJECT_SOURCE_DIR}/domain/include)
set(RYOANJI_DIR ${PROJECT_SOURCE_DIR}/ryoanji/src)

set(SPH_EXA_INCLUDE_DIRS ${CSTONE_DIR} ${RYOANJI_DIR} ${PROJECT_SOURCE_DIR}/include
                         ${PROJECT_SOURCE_DIR}/src ${MPI_CXX_INCLUDE_PATH})

add_subdirectory(evrard)
add_subdirectory(sedov)
add_subdirectory(noh)
if(BUILD_ANALYTICAL)
    add_subdirectory(analytical_solutions)
endif()
