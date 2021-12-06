
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

add_executable(ryoanji demo.cu)

target_link_libraries(ryoanji PUBLIC cstone_tree)
set_target_properties(ryoanji PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_source_files_properties(test/serial.cu PROPERTIES COMPILE_FLAGS "-Xptxas -v")