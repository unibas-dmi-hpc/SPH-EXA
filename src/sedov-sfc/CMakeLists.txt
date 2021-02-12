
add_executable(sedov-sfc sedov-sfc.cpp)
target_include_directories(sedov-sfc PRIVATE ${PROJECT_SOURCE_DIR}/domain/include)
target_include_directories(sedov-sfc PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(sedov-sfc PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(sedov-sfc ${MPI_CXX_LIBRARIES})