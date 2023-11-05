add_subdirectory(hydro_turb)

set(UNIT_TESTS
        std.cpp
        table_creation.cpp
        ve.cpp
)

set(testname sph_tests)
add_executable(${testname} ${UNIT_TESTS})
target_compile_options(${testname} PRIVATE -Wall -Wextra)

target_include_directories(${testname} PRIVATE ${CSTONE_DIR} ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(${testname} PRIVATE GTest::gtest_main)
add_test(NAME ${testname} COMMAND ${testname})

install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/hydro)
install(FILES ${PROJECT_SOURCE_DIR}/test/example_data.txt DESTINATION ${CMAKE_INSTALL_SBINDIR}/hydro)

# create a symbolic link in the build directory to the example data file
add_custom_command(TARGET ${testname} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${PROJECT_SOURCE_DIR}/test/example_data.txt $<TARGET_FILE_DIR:${testname}>/example_data.txt)