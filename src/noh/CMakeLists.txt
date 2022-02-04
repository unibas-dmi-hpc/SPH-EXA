add_executable(noh noh.cpp)

target_include_directories(noh PRIVATE ${PROJECT_SOURCE_DIR}/domain/include)
target_include_directories(noh PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(noh PRIVATE ${MPI_CXX_INCLUDE_PATH})

target_link_libraries(noh PRIVATE ${MPI_CXX_LIBRARIES})
target_link_libraries(noh PRIVATE OpenMP::OpenMP_CXX)

install(TARGETS noh RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if (SPH_EXA_WITH_H5PART)
    target_compile_definitions(noh PUBLIC SPH_EXA_HAVE_H5PART)
    target_include_directories(noh PUBLIC ${PROJECT_SOURCE_DIR}/extern/h5part)
    target_link_libraries(noh PRIVATE H5Part ${HDF5_LIBRARIES})
endif()

if(CMAKE_CUDA_COMPILER)
    add_executable(noh-cuda $<TARGET_OBJECTS:gather_obj> $<TARGET_OBJECTS:cuda_find_neighbors_obj> $<TARGET_OBJECTS:cuda_sph> noh.cpp)
    target_include_directories(noh-cuda PRIVATE  ${PROJECT_SOURCE_DIR}/domain/include)
    target_include_directories(noh-cuda PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_include_directories(noh-cuda PRIVATE ${MPI_CXX_INCLUDE_PATH})
    target_compile_definitions(noh-cuda PRIVATE USE_MPI USE_CUDA)
    set_target_properties(noh-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(noh-cuda PRIVATE ${MPI_CXX_LIBRARIES})
    target_link_libraries(noh-cuda PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(noh-cuda PRIVATE CUDA::cudart)
    install(TARGETS noh-cuda RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
