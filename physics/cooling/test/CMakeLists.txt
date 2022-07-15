project(sphexa-cooling)
set(UNIT_TESTS cooling.cpp)

set(testname cooling_tests)
add_executable(${testname} ${UNIT_TESTS})
target_compile_options(${testname} PRIVATE -Wall -Wextra)
target_compile_definitions(${testname} PUBLIC SPH_EXA_HAVE_H5PART)

target_include_directories(${testname} PRIVATE ${SPH_DIR} ${CSTONE_DIR} ${PROJECT_SOURCE_DIR}/main/src)
target_include_directories(${testname} PRIVATE ${CMAKE_SOURCE_DIR}/physics)


target_link_libraries(${testname} PRIVATE ${MPI_CXX_LIBRARIES})
target_link_libraries(${testname} PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(${testname} PRIVATE grackle)
target_link_libraries(${testname} PRIVATE GTest::gtest_main)


add_test(NAME ${testname} COMMAND ${testname})

install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/physics)
