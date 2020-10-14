
add_executable(sfc_tests clz.cpp morton.cpp octree.cpp zorder.cpp findneighbors.cpp test_main.cpp)
target_include_directories(sfc_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(sfc_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(sfc_tests gtest_main)
add_test(NAME SfcTests COMMAND sfc_tests)
