
add_executable(sedov sedov.cpp)
target_include_directories(sedov PRIVATE ${PROJECT_SOURCE_DIR}/domain/include)
target_include_directories(sedov PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(sedov PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_include_directories(sedov PRIVATE ${CUDA_INCLUDE_DIR})

target_link_libraries(sedov ${MPI_CXX_LIBRARIES})
target_link_libraries(sedov OpenMP::OpenMP_CXX)

if(CMAKE_CUDA_COMPILER)
    # this target uses cuda but isn't compiled with cuda compiler
    # so add the right cuda runtime lib
    find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

    add_executable(sedov-cuda $<TARGET_OBJECTS:gather_obj> $<TARGET_OBJECTS:cuda_find_neighbors_obj> $<TARGET_OBJECTS:cuda_sph> sedov.cpp)
    target_include_directories(sedov-cuda PRIVATE  ${PROJECT_SOURCE_DIR}/domain/include)
    target_include_directories(sedov-cuda PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_include_directories(sedov-cuda PRIVATE ${MPI_CXX_INCLUDE_PATH})
    target_include_directories(sedov-cuda PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_compile_definitions(sedov-cuda PRIVATE USE_MPI USE_CUDA)
    set_target_properties(sedov-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(sedov-cuda ${MPI_CXX_LIBRARIES})
    target_link_libraries(sedov-cuda OpenMP::OpenMP_CXX)
    target_link_libraries(sedov-cuda ${CUDART_LIBRARY})
    install(TARGETS sedov-cuda RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

install(TARGETS sedov RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
