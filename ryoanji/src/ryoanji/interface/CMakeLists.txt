if(CMAKE_HIP_COMPILER)
    set_source_files_properties(ewald.cu treebuilder.cu multipole_holder.cu PROPERTIES LANGUAGE HIP)
endif()

if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(cstone_tree treebuilder.cu)
    target_include_directories(cstone_tree PUBLIC ${PROJECT_SOURCE_DIR}/src)
    target_include_directories(cstone_tree PUBLIC ${CSTONE_DIR})
    target_link_libraries(cstone_tree cstone_gpu)

    # workaround for compiling MPI with nvcc on cray systems
    if (MPI_CXX_INCLUDE_PATH STREQUAL "" AND DEFINED ENV{CRAY_MPICH2_DIR} OR DEFINED ENV{CRAY_MPICH_DIR})
        message(STATUS "Applying MPI include workaround on CRAY")
        set(MPI_CXX_INCLUDE_PATH $ENV{MPICH_DIR}/include)
    endif()
    message(STATUS "MPI include is " ${MPI_CXX_INCLUDE_PATH})

    add_library(ryoanji ewald.cu multipole_holder.cu)
    target_include_directories(ryoanji PUBLIC ${PROJECT_SOURCE_DIR}/src ${CSTONE_DIR} ${MPI_CXX_INCLUDE_PATH})
    target_link_libraries(ryoanji PUBLIC cstone_gpu ${MPI_CXX_LIBRARIES})
endif()
