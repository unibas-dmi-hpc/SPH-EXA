
set(CSTONE_DIR ${CMAKE_SOURCE_DIR}/domain/include)
set(CSTONE_TEST_DIR ${CMAKE_SOURCE_DIR}/domain/test)

set(RYOANJI_TEST_INCLUDE_DIRS ${CSTONE_DIR} ${CSTONE_TEST_DIR} ${PROJECT_SOURCE_DIR}/src)

add_executable(ryoanji_cpu_unit_tests
               multipole.cpp
               treewalk.cpp
               ../test_main.cpp)

target_compile_options(ryoanji_cpu_unit_tests PRIVATE -Wall -Wextra -Wno-unknown-pragmas)

target_include_directories(ryoanji_cpu_unit_tests PRIVATE ${RYOANJI_TEST_INCLUDE_DIRS})

target_link_libraries(ryoanji_cpu_unit_tests PRIVATE GTest::gtest_main)
add_test(NAME RyoanjiCpuTests COMMAND ryoanji_cpu_unit_tests)

install(TARGETS ryoanji_unit_tests RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/ryoanji/cpu_unit_tests)


add_executable(gravity_perf gravity.cpp)
target_include_directories(gravity_perf PRIVATE ${RYOANJI_TEST_INCLUDE_DIRS})
target_link_libraries(gravity_perf PRIVATE OpenMP::OpenMP_CXX)
#target_compile_options(gravity_perf PRIVATE -Rpass=loop-vectorize -Rpass-analysis=loop-vectorize)
install(TARGETS gravity_perf RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/performance)

add_executable(multi_rank multi_rank.cpp)
target_include_directories(multi_rank PRIVATE ${MPI_CXX_INCLUDE_PATH} ${RYOANJI_TEST_INCLUDE_DIRS})
target_link_libraries(multi_rank PRIVATE OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES})
