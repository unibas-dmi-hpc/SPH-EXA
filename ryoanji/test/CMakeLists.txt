
add_subdirectory(gtest)

add_executable(ryoanji_unit_tests
        treebuilder.cu
        direct.cu
        multipole.cu
        warpscan.cu
        test_main.cpp)

target_include_directories(ryoanji_unit_tests PRIVATE ../src)

target_link_libraries(ryoanji_unit_tests PUBLIC OpenMP::OpenMP_CXX cstone_tree gtest_main)

set_source_files_properties(buildtree.cu PROPERTIES COMPILE_FLAGS -G)
set_source_files_properties(direct.cu PROPERTIES COMPILE_FLAGS -G)
set_source_files_properties(multipole.cu PROPERTIES COMPILE_FLAGS -G)

install(TARGETS ryoanji_unit_tests RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/ryoanji/unit_tests)
