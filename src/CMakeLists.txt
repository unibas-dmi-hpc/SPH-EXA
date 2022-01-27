
add_subdirectory(evrard)
add_subdirectory(sedov)
if(BUILD_ANALYTICAL)
    add_subdirectory(analytical_solutions)
endif()
