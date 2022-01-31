
add_subdirectory(evrard)
add_subdirectory(sedov)
add_subdirectory(noh)
if(BUILD_ANALYTICAL)
    add_subdirectory(analytical_solutions)
endif()
