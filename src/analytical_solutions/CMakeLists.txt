
add_subdirectory(sedov_solution)

install(FILES ${CMAKE_SOURCE_DIR}/src/analytical_solutions/compare_solutions.py TYPE BIN)
