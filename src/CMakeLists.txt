
add_subdirectory(evrard)
add_subdirectory(sedov)
if(BUILD_ANALYTICAL)
    add_subdirectory(analyticalSolutions)
endif()
