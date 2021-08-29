
add_subdirectory(gtest)

add_executable(ryoanji_unit_tests
        direct.cu
        multipole.cu
        test_main.cpp)

target_include_directories(ryoanji_unit_tests PRIVATE ../include)
target_include_directories(ryoanji_unit_tests PRIVATE ../../domain/include)
target_link_libraries(ryoanji_unit_tests PUBLIC OpenMP::OpenMP_CXX gtest_main)
set_target_properties(ryoanji_unit_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS ryoanji_unit_tests RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/ryoanji/unit_tests)
