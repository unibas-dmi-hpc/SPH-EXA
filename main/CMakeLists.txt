
set(CSTONE_DIR ${PROJECT_SOURCE_DIR}/domain/include)
set(RYOANJI_DIR ${PROJECT_SOURCE_DIR}/ryoanji/src)
set(SPH_DIR ${PROJECT_SOURCE_DIR}/sph/include)

add_subdirectory(src)
if (BUILD_TESTING)
    add_subdirectory(test)
endif (BUILD_TESTING)