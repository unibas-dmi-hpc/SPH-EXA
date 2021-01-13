
add_executable(sfc_tests
        boxoverlap.cpp
        btree.cpp
        btreetraversal.cpp
        collisions_cpu.cpp
        clz.cpp
        domaindecomp.cpp
        disjointoctree.cpp
        findneighbors.cpp
        halodiscovery.cpp
        haloneighbors.cpp
        histogram.cpp
        layout.cpp
        morton.cpp
        octree.cpp
        octree_util.cpp
        zorder.cpp
        test_main.cpp)

target_include_directories(sfc_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(sfc_tests PRIVATE ${PROJECT_SOURCE_DIR}/test/unit)
target_link_libraries(sfc_tests gtest_main)
add_test(NAME SfcTests COMMAND sfc_tests)
