
add_executable(component_units
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

target_include_directories(component_units PRIVATE ../../include)
target_include_directories(component_units PRIVATE ../)
target_link_libraries(component_units gtest_main)
add_test(NAME ComponentUnits COMMAND component_units)
