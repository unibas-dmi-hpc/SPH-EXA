
# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

add_executable(run_mpi_tests test_main.cpp)
target_include_directories(run_mpi_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(run_mpi_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(run_mpi_tests PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(run_mpi_tests gtest_main)
target_link_libraries(run_mpi_tests ${MPI_CXX_LIBRARIES})

add_executable(run_sfc_tests sfc_tests.cpp)
target_include_directories(run_sfc_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(run_sfc_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(run_sfc_tests gtest_main)
add_test(NAME run_sfc_tests COMMAND run_sfc_tests.exe)
