add_executable(sedov sedov.cpp)
target_include_directories(sedov PRIVATE ${PROJECT_SOURCE_DIR}/domain/include)
target_include_directories(sedov PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(sedov PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(sedov PRIVATE ${MPI_CXX_LIBRARIES})
target_link_libraries(sedov PRIVATE OpenMP::OpenMP_CXX)
install(TARGETS sedov RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
if(BUILD_ANALYTICAL)
install(FILES ${CMAKE_SOURCE_DIR}/src/analyticalSolutions/compare_solutions.py
        TYPE SBIN)
endif()

# {{{ i/o
if (SPH_EXA_WITH_H5PART)
    target_compile_definitions(sedov PUBLIC SPH_EXA_HAVE_H5PART)
    target_include_directories(sedov PUBLIC ${PROJECT_SOURCE_DIR}/extern/h5part)
    target_link_libraries(sedov PRIVATE H5Part ${HDF5_LIBRARIES})
endif()

if(INSITU STREQUAL "Catalyst")
    target_include_directories(sedov PRIVATE ..)
    target_compile_definitions(sedov
	    PRIVATE
	    "PARAVIEW_IMPL_DIR=\"${ParaView_CATALYST_DIR}\""
	    SPH_EXA_USE_CATALYST2)
    target_link_libraries(sedov PRIVATE catalyst::catalyst)
elseif(INSITU STREQUAL "Ascent")
    target_include_directories(sedov PRIVATE ..)
    target_compile_definitions(sedov
	    PRIVATE
	    SPH_EXA_USE_ASCENT)
    target_link_libraries(sedov PRIVATE ascent::ascent_mpi)
endif()
# }}}

if(CMAKE_CUDA_COMPILER)
    add_executable(sedov-cuda $<TARGET_OBJECTS:gather_obj> $<TARGET_OBJECTS:cuda_find_neighbors_obj> $<TARGET_OBJECTS:cuda_sph> sedov.cpp)
    target_include_directories(sedov-cuda PRIVATE  ${PROJECT_SOURCE_DIR}/domain/include)
    target_include_directories(sedov-cuda PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_include_directories(sedov-cuda PRIVATE ${MPI_CXX_INCLUDE_PATH})
    target_compile_definitions(sedov-cuda PRIVATE USE_MPI USE_CUDA)
    set_target_properties(sedov-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(sedov-cuda PRIVATE ${MPI_CXX_LIBRARIES})
    target_link_libraries(sedov-cuda PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(sedov-cuda PRIVATE CUDA::cudart)
    install(TARGETS sedov-cuda RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

elseif(CMAKE_HIP_COMPILER)
    set_source_files_properties(sedov.cpp PROPERTIES LANGUAGE HIP)
    add_executable(sedov-hip $<TARGET_OBJECTS:gather_obj> $<TARGET_OBJECTS:cuda_find_neighbors_obj> $<TARGET_OBJECTS:cuda_sph> sedov.cpp)
    set_property(TARGET sedov-hip PROPERTY HIP_ARCHITECTURES gfx908)
    target_include_directories(sedov-hip PRIVATE ${PROJECT_SOURCE_DIR}/domain/include)
    target_include_directories(sedov-hip PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_include_directories(sedov-hip PRIVATE ${MPI_CXX_INCLUDE_PATH})
    target_include_directories(sedov-hip PRIVATE ${HIP_PATH}/include)
    target_compile_definitions(sedov-hip PRIVATE USE_MPI USE_CUDA)
    target_link_libraries(sedov-hip PRIVATE ${MPI_CXX_LIBRARIES})
    target_link_libraries(sedov-hip PRIVATE OpenMP::OpenMP_CXX)
    install(TARGETS sedov-hip RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
