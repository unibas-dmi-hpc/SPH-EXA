
function(enableH5Part exename)
    if (SPH_EXA_WITH_H5PART)
        target_compile_definitions(${exename} PUBLIC SPH_EXA_HAVE_H5PART)
        target_include_directories(${exename} PUBLIC ${PROJECT_SOURCE_DIR}/extern/h5part)
        target_link_libraries(${exename} PRIVATE H5Part ${HDF5_LIBRARIES})
    endif()
endfunction()

add_executable(sedov sedov.cpp)
target_include_directories(sedov PRIVATE ${SPH_EXA_INCLUDE_DIRS})
target_link_libraries(sedov PRIVATE ${MPI_CXX_LIBRARIES})
target_link_libraries(sedov PRIVATE OpenMP::OpenMP_CXX)
enableH5Part(sedov)
install(TARGETS sedov RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# {{{ i/o
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
    target_include_directories(sedov-cuda PRIVATE ${SPH_EXA_INCLUDE_DIRS})
    target_compile_definitions(sedov-cuda PRIVATE USE_MPI USE_CUDA)
    set_target_properties(sedov-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(sedov-cuda PRIVATE ${MPI_CXX_LIBRARIES})
    target_link_libraries(sedov-cuda PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(sedov-cuda PRIVATE CUDA::cudart)
    enableH5Part(sedov-cuda)
    install(TARGETS sedov-cuda RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

elseif(CMAKE_HIP_COMPILER)
    set_source_files_properties(sedov.cpp PROPERTIES LANGUAGE HIP)
    add_executable(sedov-hip $<TARGET_OBJECTS:gather_obj> $<TARGET_OBJECTS:cuda_find_neighbors_obj> $<TARGET_OBJECTS:cuda_sph> sedov.cpp)
    set_property(TARGET sedov-hip PROPERTY HIP_ARCHITECTURES gfx908)
    target_include_directories(sedov-hip PRIVATE ${SPH_EXA_INCLUDE_DIRS})
    target_include_directories(sedov-hip PRIVATE ${HIP_PATH}/include)
    target_compile_definitions(sedov-hip PRIVATE USE_MPI USE_CUDA)
    target_link_libraries(sedov-hip PRIVATE ${MPI_CXX_LIBRARIES})
    target_link_libraries(sedov-hip PRIVATE OpenMP::OpenMP_CXX)
    enableH5Part(sedov-hip)
    install(TARGETS sedov-hip RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
