
set(UNIT_TESTS
        domain/domaindecomp.cpp
        domain/layout.cpp
        domain/peers.cpp
        findneighbors.cpp
        halos/boxoverlap.cpp
        halos/btreetraversal.cpp
        halos/btreetraversal_a2a.cpp
        halos/discovery.cpp
        primitives/clz.cpp
        primitives/gather.cpp
        sfc/box.cpp
        sfc/common.cpp
        sfc/hilbert.cpp
        sfc/morton.cpp
        sfc/sfc.cpp
        tree/btree.cpp
        tree/macs.cpp
        tree/octree.cpp
        tree/octree_focus.cpp
        tree/octree_internal.cpp
        tree/octree_util.cpp
        tree/traversal.cpp
        tree/upsweep.cpp
        test_main.cpp)

add_executable(component_units ${UNIT_TESTS})

target_include_directories(component_units PRIVATE ../../include)
target_include_directories(component_units PRIVATE ../)
target_link_libraries(component_units PRIVATE gtest_main)
add_test(NAME ComponentUnits COMMAND component_units)

add_executable(component_units_omp ${UNIT_TESTS})

target_include_directories(component_units_omp PRIVATE ../../include)
target_include_directories(component_units_omp PRIVATE ../)
target_link_libraries(component_units_omp PRIVATE gtest_main)
target_link_libraries(component_units_omp PUBLIC OpenMP::OpenMP_CXX)
add_test(NAME ComponentUnitsOmp COMMAND component_units_omp)
