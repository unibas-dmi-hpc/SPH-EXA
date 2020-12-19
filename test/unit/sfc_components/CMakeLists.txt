
add_executable(sfc_tests
        clz.cpp
        morton.cpp
        disjointoctree.cpp
        octree.cpp
        octree_util.cpp
        zorder.cpp
        findneighbors.cpp
        domaindecomp.cpp
        histogram.cpp
        halos.cpp
        binarytree.cpp
        test_main.cpp)

target_include_directories(sfc_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(sfc_tests PRIVATE ${PROJECT_SOURCE_DIR}/test/unit)
target_link_libraries(sfc_tests gtest_main)
add_test(NAME SfcTests COMMAND sfc_tests)
