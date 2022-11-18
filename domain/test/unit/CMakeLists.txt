
set(UNIT_TESTS
        domain/domaindecomp.cpp
        domain/index_ranges.cpp
        domain/layout.cpp
        focus/octree_focus.cpp
        focus/source_center.cpp
        neighbors/findneighbors.cpp
        neighbors/neighbors_traversal.cpp
        primitives/clz.cpp
        primitives/gather.cpp
        sfc/box.cpp
        sfc/common.cpp
        sfc/hilbert.cpp
        sfc/morton.cpp
        sfc/sfc.cpp
        traversal/boxoverlap.cpp
        traversal/collisions.cpp
        traversal/collisions_a2a.cpp
        traversal/discovery.cpp
        traversal/macs.cpp
        traversal/peers.cpp
        traversal/traversal.cpp
        tree/btree.cpp
        tree/csarray.cpp
        tree/octree_internal.cpp
        tree/octree_util.cpp
        util/array.cpp
        util/util.cpp
        test_main.cpp)

add_executable(component_units ${UNIT_TESTS} neighbors/all_to_all.hpp neighbors/neighbors_traversal.cpp)
target_compile_options(component_units PRIVATE -Wall -Wextra -Wno-unknown-pragmas)

target_include_directories(component_units PRIVATE ../../include)
target_include_directories(component_units PRIVATE ../)
target_link_libraries(component_units PRIVATE GTest::gtest_main)
add_test(NAME ComponentUnits COMMAND component_units)

add_executable(component_units_omp ${UNIT_TESTS})

target_include_directories(component_units_omp PRIVATE ../../include)
target_include_directories(component_units_omp PRIVATE ../)
target_link_libraries(component_units_omp PRIVATE GTest::gtest_main)
target_link_libraries(component_units_omp PUBLIC OpenMP::OpenMP_CXX)
add_test(NAME ComponentUnitsOmp COMMAND component_units_omp)

install(TARGETS component_units RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/unit)
install(TARGETS component_units_omp RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/unit)
