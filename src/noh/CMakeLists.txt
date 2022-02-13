add_executable(noh noh.cpp)

target_include_directories(noh PRIVATE ${PROJECT_SOURCE_DIR}/domain/include)
target_include_directories(noh PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(noh PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_include_directories(noh PRIVATE ${MPI_CXX_INCLUDE_PATH})

target_link_libraries(noh PRIVATE ${MPI_CXX_LIBRARIES})
target_link_libraries(noh PRIVATE OpenMP::OpenMP_CXX)

install(TARGETS noh RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# {{{ i/o
if (SPH_EXA_WITH_H5PART)
    target_compile_definitions(noh PUBLIC SPH_EXA_HAVE_H5PART)
    target_include_directories(noh PUBLIC ${PROJECT_SOURCE_DIR}/extern/h5part)
    target_link_libraries(noh PRIVATE H5Part ${HDF5_LIBRARIES})
endif()

if(INSITU STREQUAL "Catalyst")
    target_include_directories(noh PRIVATE ..)
    target_compile_definitions(noh
	    PRIVATE
	    "PARAVIEW_IMPL_DIR=\"${ParaView_CATALYST_DIR}\""
	    SPH_EXA_USE_CATALYST2)
    target_link_libraries(noh PRIVATE catalyst::catalyst)
elseif(INSITU STREQUAL "Ascent")
    target_include_directories(noh PRIVATE ..)
    target_compile_definitions(noh
	    PRIVATE
	    SPH_EXA_USE_ASCENT)
    target_link_libraries(noh PRIVATE ascent::ascent_mpi)
endif()
# }}}

if(CMAKE_CUDA_COMPILER)
    add_executable(noh-cuda $<TARGET_OBJECTS:gather_obj> $<TARGET_OBJECTS:cuda_find_neighbors_obj> $<TARGET_OBJECTS:cuda_sph> noh.cpp)
    target_include_directories(noh-cuda PRIVATE  ${PROJECT_SOURCE_DIR}/domain/include)
    target_include_directories(noh-cuda PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_include_directories(noh-cuda PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_include_directories(noh-cuda PRIVATE ${MPI_CXX_INCLUDE_PATH})
    target_compile_definitions(noh-cuda PRIVATE USE_MPI USE_CUDA)
    set_target_properties(noh-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(noh-cuda PRIVATE ${MPI_CXX_LIBRARIES})
    target_link_libraries(noh-cuda PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(noh-cuda PRIVATE CUDA::cudart)
    install(TARGETS noh-cuda RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

elseif(CMAKE_HIP_COMPILER)
    set_source_files_properties(noh.cpp PROPERTIES LANGUAGE HIP)
    add_executable(noh-hip $<TARGET_OBJECTS:gather_obj> $<TARGET_OBJECTS:cuda_find_neighbors_obj> $<TARGET_OBJECTS:cuda_sph> noh.cpp)
    set_property(TARGET noh-hip PROPERTY HIP_ARCHITECTURES gfx908)
    target_include_directories(noh-hip PRIVATE ${PROJECT_SOURCE_DIR}/domain/include)
    target_include_directories(noh-hip PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_include_directories(noh-hip PRIVATE ${PROJECT_SOURCE_DIR}/src)
    target_include_directories(noh-hip PRIVATE ${MPI_CXX_INCLUDE_PATH})
    target_include_directories(noh-hip PRIVATE ${HIP_PATH}/include)
    target_compile_definitions(noh-hip PRIVATE USE_MPI USE_CUDA)
    target_link_libraries(noh-hip PRIVATE ${MPI_CXX_LIBRARIES})
    target_link_libraries(noh-hip PRIVATE OpenMP::OpenMP_CXX)
    install(TARGETS noh-hip RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
