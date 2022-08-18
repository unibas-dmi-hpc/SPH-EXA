
set(testname global_upsweep_cpu)
add_executable(${testname} global_upsweep_cpu.cpp)
target_include_directories(${testname} PRIVATE ${MPI_CXX_INCLUDE_PATH} ${RYOANJI_TEST_INCLUDE_DIRS})
target_link_libraries(${testname} PRIVATE ${MPI_CXX_LIBRARIES})
target_compile_options(${testname} PRIVATE -Wno-unknown-pragmas)
ryoanji_add_test(${testname} EXECUTABLE ${testname} RANKS 10)
install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/ryoanji/)
unset(testname)

if (CMAKE_HIP_COMPILER)
    set_source_files_properties(global_upsweep_gpu.cpp PROPERTIES LANGUAGE HIP)
endif()

if (CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    set(testname global_upsweep_gpu)
    add_executable(${testname} global_upsweep_gpu.cpp)
    target_include_directories(${testname} PRIVATE ${RYOANJI_TEST_INCLUDE_DIRS})

    target_link_libraries(${testname} PRIVATE ryoanji)

    if (CMAKE_HIP_COMPILER)
        # we don't need HDF5, but linking against it pulls in the link flags for MPI librarys
        # cmake together with the cray compiler wrapper screws up here and does not properly link
        # HIP translation units againts MPI
        target_link_libraries(${testname} PRIVATE H5Part ${HDF5_LIBRARIES})
    endif ()

    if (CMAKE_CUDA_COMPILER)
        target_link_libraries(${testname} PRIVATE CUDA::cudart)
    endif ()
    target_compile_options(${testname} PRIVATE -Wno-unknown-pragmas)
    ryoanji_add_test(${testname} EXECUTABLE ${testname} RANKS 10)
    install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/ryoanji/)
    unset(testname)
endif ()
