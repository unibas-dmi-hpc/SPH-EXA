if (SPH_EXA_WITH_NNET)
    set(SPH_DIR ${CMAKE_SOURCE_DIR}/sph/include)
    set(CSTONE_DIR ${CMAKE_SOURCE_DIR}/domain/include)

    add_subdirectory(nuclear-networks)
endif()

if (SPH_EXA_WITH_GRACKLE)
    add_subdirectory(cooling)
endif()