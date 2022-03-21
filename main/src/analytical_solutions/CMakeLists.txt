
add_subdirectory(sedov_solution)

install(FILES ${CMAKE_SOURCE_DIR}/main/src/analytical_solutions/compare_solutions.py TYPE BIN)
