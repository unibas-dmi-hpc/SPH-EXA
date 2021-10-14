
add_executable(evrard evrard.cpp)
target_include_directories(evrard PRIVATE ${PROJECT_SOURCE_DIR}/domain/include)
target_include_directories(evrard PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(evrard PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_compile_definitions(evrard PRIVATE USE_MPI GRAVITY)
target_link_libraries(evrard ${MPI_CXX_LIBRARIES})
target_link_libraries(evrard OpenMP::OpenMP_CXX)

install(TARGETS evrard RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
