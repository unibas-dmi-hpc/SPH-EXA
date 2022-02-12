
set(exename evrard)
add_executable(${exename} evrard.cpp)

target_include_directories(${exename} PRIVATE ${SPH_EXA_INCLUDE_DIRS})
target_compile_definitions(${exename} PRIVATE USE_MPI GRAVITY)

target_link_libraries(${exename} PRIVATE ${MPI_CXX_LIBRARIES})
target_link_libraries(${exename} PRIVATE OpenMP::OpenMP_CXX)

if (SPH_EXA_WITH_H5PART)
    target_include_directories(${exename} PUBLIC ${PROJECT_SOURCE_DIR}/extern/h5part)
    target_compile_definitions(${exename} PUBLIC SPH_EXA_HAVE_H5PART)
    target_link_libraries(${exename} PRIVATE H5Part ${HDF5_LIBRARIES})
endif()

if (SPH_EXA_WITH_CATALYST2)
    target_include_directories(${exename} PRIVATE ..)
    target_compile_definitions(${exename}
	    PRIVATE
	    "PARAVIEW_IMPL_DIR=\"${ParaView_CATALYST_DIR}\""
	    SPH_EXA_USE_CATALYST2)
    target_link_libraries(${exename} PRIVATE catalyst::catalyst)
endif()

install(TARGETS ${exename} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
unset(exename)
