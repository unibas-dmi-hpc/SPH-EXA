
<<<<<<< HEAD
add_executable(sedov sedov.cpp)
target_include_directories(sedov PRIVATE ${PROJECT_SOURCE_DIR}/domain/include)
target_include_directories(sedov PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(sedov PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(sedov ${MPI_CXX_LIBRARIES})
target_link_libraries(sedov OpenMP::OpenMP_CXX)
target_link_libraries(sedov CUDA::cudart)

add_executable(sedov-cuda $<TARGET_OBJECTS:gather_obj> $<TARGET_OBJECTS:cuda_find_neighbors_obj> $<TARGET_OBJECTS:cuda_sph> sedov.cpp)
target_include_directories(sedov-cuda PRIVATE  ${PROJECT_SOURCE_DIR}/domain/include)
target_include_directories(sedov-cuda PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(sedov-cuda PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_compile_definitions(sedov-cuda PRIVATE USE_MPI USE_CUDA)
set_target_properties(sedov-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(sedov-cuda ${MPI_CXX_LIBRARIES})
target_link_libraries(sedov-cuda OpenMP::OpenMP_CXX)
target_link_libraries(sedov-cuda CUDA::cudart)
=======
add_executable(sedov-sfc sedov-sfc.cpp)
target_include_directories(sedov-sfc PRIVATE ${PROJECT_SOURCE_DIR}/domain/include)
target_include_directories(sedov-sfc PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(sedov-sfc PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(sedov-sfc ${MPI_CXX_LIBRARIES})
>>>>>>> added missing includes
