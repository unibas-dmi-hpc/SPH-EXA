
set(exename evrard_new)
add_executable(${exename} evrard_new.cpp)

target_include_directories(${exename} PRIVATE ${PROJECT_SOURCE_DIR}/domain/include)
target_include_directories(${exename} PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(${exename} PRIVATE ${MPI_CXX_INCLUDE_PATH})

target_compile_definitions(${exename} PRIVATE USE_MPI GRAVITY)

target_link_libraries(${exename} PRIVATE ${MPI_CXX_LIBRARIES})
target_link_libraries(${exename} PRIVATE OpenMP::OpenMP_CXX)

if (SPH_EXA_WITH_H5PART)
    target_include_directories(${exename} PUBLIC ${PROJECT_SOURCE_DIR}/extern/h5part)
    target_compile_definitions(${exename} PUBLIC SPH_EXA_HAVE_H5PART)
    target_link_libraries(${exename} PRIVATE H5Part ${HDF5_LIBRARIES})
endif()

install(TARGETS ${exename} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
unset(exename)
