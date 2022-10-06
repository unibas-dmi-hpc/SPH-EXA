
function(enableH5Part exename)
    if (SPH_EXA_WITH_H5PART)
        target_compile_definitions(${exename} PUBLIC SPH_EXA_HAVE_H5PART)
        target_include_directories(${exename} PUBLIC ${PROJECT_SOURCE_DIR}/extern/h5part)
        # necessary because the HDF5_INLUCDE_DIRS is empty on CRAY and HIP-compilation is done without the cray-wrapper
        target_include_directories(${exename} PUBLIC $ENV{HDF5_DIR}/include)
        target_link_libraries(${exename} PRIVATE H5Part ${HDF5_LIBRARIES})
    endif()
endfunction()

function(enableInSituViz exename)
  if(INSITU STREQUAL "Catalyst")
    target_include_directories(${exename} PRIVATE ..)
    target_compile_definitions(${exename}
            PRIVATE
            "PARAVIEW_IMPL_DIR=\"${ParaView_CATALYST_DIR}\""
            SPH_EXA_USE_CATALYST2)
    target_link_libraries(${exename} PRIVATE catalyst::catalyst)
  elseif(INSITU STREQUAL "Ascent")
    target_include_directories(${exename} PRIVATE ..)
    target_compile_definitions(${exename}
            PRIVATE
            SPH_EXA_USE_ASCENT)
    target_link_libraries(${exename} PRIVATE ascent::ascent_mpi)
  endif()
endfunction()

function(enableGpuDirect exename)
    if (GPU_DIRECT)
        target_compile_definitions(${exename} PRIVATE USE_GPU_DIRECT)
    endif()
endfunction()

set(exename sphexa)
add_executable(${exename} sphexa.cpp)
target_include_directories(${exename} PRIVATE ${SPH_EXA_INCLUDE_DIRS})
target_link_libraries(${exename} PRIVATE OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES})
enableH5Part(${exename})
enableInSituViz(${exename})
install(TARGETS ${exename} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if (CMAKE_CUDA_COMPILER)
    add_executable(${exename}-cuda sphexa.cpp)
    target_include_directories(${exename}-cuda PRIVATE ${SPH_EXA_INCLUDE_DIRS})
    target_include_directories(${exename}-cuda PUBLIC ${CMAKE_BINARY_DIR}/main/src)
    target_compile_definitions(${exename}-cuda PRIVATE USE_MPI USE_CUDA)
    target_link_libraries(${exename}-cuda PRIVATE cstone_gpu ryoanji sph_gpu observables_gpu OpenMP::OpenMP_CXX
                          ${MPI_CXX_LIBRARIES} CUDA::cudart)
    enableH5Part(${exename}-cuda)
    enableInSituViz(${exename}-cuda)
    enableGpuDirect(${exename}-cuda)
    install(TARGETS ${exename}-cuda RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

elseif (CMAKE_HIP_COMPILER)
    add_executable(${exename}-hip sphexa.cpp)
    target_include_directories(${exename}-hip PRIVATE ${SPH_EXA_INCLUDE_DIRS})
    target_compile_definitions(${exename}-hip PRIVATE USE_MPI USE_CUDA THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)
    target_link_libraries(${exename}-hip PRIVATE cstone_gpu ryoanji sph_gpu observables_gpu OpenMP::OpenMP_CXX
                          ${MPI_CXX_LIBRARIES} hip::host)
    set_target_properties(${exename}-hip PROPERTIES LINKER_LANGUAGE CXX)
    enableH5Part(${exename}-hip)
    enableInSituViz(${exename}-hip)
    enableGpuDirect(${exename}-hip)
    install(TARGETS ${exename}-hip RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif ()
