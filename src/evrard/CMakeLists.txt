# this target uses cuda but isn't compiled with cuda compiler
# so add the right cuda runtime lib
find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

add_executable(evrard evrard.cpp)
target_include_directories(evrard PRIVATE ${PROJECT_SOURCE_DIR}/domain/include)
target_include_directories(evrard PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(evrard PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_include_directories(evrard PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_compile_definitions(evrard PRIVATE USE_MPI GRAVITY)
target_link_libraries(evrard ${MPI_CXX_LIBRARIES})
target_link_libraries(evrard OpenMP::OpenMP_CXX)
target_link_libraries(evrard ${CUDART_LIBRARY})

install(TARGETS evrard RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
