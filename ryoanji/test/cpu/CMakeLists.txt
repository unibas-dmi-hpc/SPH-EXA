include(ryoanji_add_test)

set(CSTONE_DIR ${CMAKE_SOURCE_DIR}/domain/include)
set(CSTONE_TEST_DIR ${CMAKE_SOURCE_DIR}/domain/test)

set(RYOANJI_TEST_INCLUDE_DIRS ${CSTONE_DIR} ${CSTONE_TEST_DIR} ${PROJECT_SOURCE_DIR}/src)

add_executable(ryoanji_cpu_unit_tests
               multipole.cpp
               treewalk.cpp
               ../test_main.cpp)

target_compile_options(ryoanji_cpu_unit_tests PRIVATE -Wall -Wextra -Wno-unknown-pragmas)

target_include_directories(ryoanji_cpu_unit_tests PRIVATE ${RYOANJI_TEST_INCLUDE_DIRS})

target_link_libraries(ryoanji_cpu_unit_tests PRIVATE GTest::gtest_main OpenMP::OpenMP_CXX)
add_test(NAME ryoanji_cpu_unit_tests COMMAND ryoanji_cpu_unit_tests)

install(TARGETS ryoanji_cpu_unit_tests RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/ryoanji/cpu_unit_tests)

set(testname multi_rank)
add_executable(${testname} multi_rank.cpp)
target_include_directories(${testname} PRIVATE ${MPI_CXX_INCLUDE_PATH} ${RYOANJI_TEST_INCLUDE_DIRS})
target_link_libraries(${testname} PRIVATE OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES})
ryoanji_add_test(${testname} EXECUTABLE ${testname} RANKS 2)
install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/ryoanji)
unset(testname)
