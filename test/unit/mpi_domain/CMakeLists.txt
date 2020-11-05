
add_executable(globaloctree globaloctree.cpp test_main.cpp)
target_include_directories(globaloctree PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(globaloctree PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(globaloctree ${MPI_CXX_LIBRARIES} gtest_main)
add_test(NAME GlobalTreeTests COMMAND globaloctree)

add_executable(exchangeparticles exchangeparticles.cpp test_main.cpp)
target_include_directories(exchangeparticles PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(exchangeparticles PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(exchangeparticles ${MPI_CXX_LIBRARIES} gtest_main)
add_test(NAME GlobalDomainExchange COMMAND exchangeparticles)

add_executable(treedomain treedomain.cpp test_main.cpp)
target_include_directories(treedomain PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(treedomain PRIVATE ${PROJECT_SOURCE_DIR}/test/unit)
target_include_directories(treedomain PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(treedomain ${MPI_CXX_LIBRARIES} gtest_main)
add_test(NAME GlobalDomainTreeIntegration COMMAND treedomain)
