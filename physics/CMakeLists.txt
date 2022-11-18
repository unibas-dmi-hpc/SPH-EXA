if (SPH_EXA_WITH_NNET)
    add_subdirectory(nuclear_networks)
endif()

if (SPH_EXA_WITH_GRACKLE)
    add_subdirectory(cooling)
endif()