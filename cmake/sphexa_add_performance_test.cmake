include(sphexa_add_test)

function(sphexa_add_performance_test sourcename testname ranks)
    add_executable(${testname} ${sourcename})
    target_include_directories(${testname} PRIVATE ../../include)
    target_include_directories(${testname} PRIVATE ../)
    target_link_libraries(${testname} PRIVATE OpenMP::OpenMP_CXX)
    install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/performance)
endfunction()

function(sphexa_add_cuda_performance_test sourcename objectname testname ranks)
    add_executable(${testname} ${objectname} ${sourcename})
    target_include_directories(${testname} PRIVATE ../../include)
    target_include_directories(${testname} PRIVATE ../)
    target_link_libraries(${testname} PRIVATE ${CUDA_RUNTIME_LIBRARY} OpenMP::OpenMP_CXX)
    install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/performance)
endfunction()
