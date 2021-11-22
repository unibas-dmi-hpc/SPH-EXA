
set(UNIT_TESTS
        domain/domaindecomp.cpp
        domain/layout.cpp
        findneighbors.cpp
        focus/octree_focus.cpp
        gravity/multipole.cpp
        gravity/treewalk.cpp
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
        traversal/upsweep.cpp
        tree/btree.cpp
        tree/octree.cpp
        tree/octree_internal.cpp
        tree/octree_internal_td.cpp
        tree/octree_util.cpp
        test_main.cpp)

add_executable(component_units ${UNIT_TESTS})
target_compile_options(component_units PRIVATE -Wall -Wextra)

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

install(TARGETS component_units RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/unit)
install(TARGETS component_units_omp RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/unit)
