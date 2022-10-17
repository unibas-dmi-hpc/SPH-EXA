if (SPH_EXA_WITH_NNET)
    add_subdirectory(nuclear-networks)
endif()

if (SPH_EXA_WITH_GRACKLE)
    add_subdirectory(cooling)
endif()