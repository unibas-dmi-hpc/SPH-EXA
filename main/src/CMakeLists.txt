
set(CSTONE_DIR ${PROJECT_SOURCE_DIR}/domain/include)
set(RYOANJI_DIR ${PROJECT_SOURCE_DIR}/ryoanji/src)
set(SPH_DIR ${PROJECT_SOURCE_DIR}/sph/include)

set(SPH_EXA_INCLUDE_DIRS ${CSTONE_DIR} ${RYOANJI_DIR} ${SPH_DIR}
        ${PROJECT_SOURCE_DIR}/main/src ${MPI_CXX_INCLUDE_PATH})

function(enableH5Part exename)
    if (SPH_EXA_WITH_H5PART)
        target_compile_definitions(${exename} PUBLIC SPH_EXA_HAVE_H5PART)
        target_include_directories(${exename} PUBLIC ${PROJECT_SOURCE_DIR}/extern/h5part)
        target_link_libraries(${exename} PRIVATE H5Part ${HDF5_LIBRARIES})
    endif ()
endfunction()


add_subdirectory(sphexa)
if (BUILD_ANALYTICAL)
    add_subdirectory(analytical_solutions)
endif ()
