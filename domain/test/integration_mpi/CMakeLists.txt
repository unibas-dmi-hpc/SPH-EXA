include(cstone_add_test)

function(addMpiTest source exename testname ranks)
    add_executable(${exename} ${source} test_main.cpp)
    target_include_directories(${exename} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_include_directories(${exename} PRIVATE ${PROJECT_SOURCE_DIR}/test)
    target_include_directories(${exename} PRIVATE ${MPI_CXX_INCLUDE_PATH})
    target_compile_options(${exename} PRIVATE -Wno-unknown-pragmas)
    target_link_libraries(${exename} ${MPI_CXX_LIBRARIES} GTest::gtest_main)
    cstone_add_test(${testname} EXECUTABLE ${exename} RANKS ${ranks})
    install(TARGETS ${exename} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/integration_mpi)
endfunction()

addMpiTest(globaloctree.cpp globaloctree GlobalTreeTests 2)
addMpiTest(exchange_halos.cpp exchange_halos GlobalHaloExchange 2)
addMpiTest(box_mpi.cpp box_mpi GlobalBox 2)
addMpiTest(exchange_focus.cpp exchange_focus GlobalFocusExchange 2)
addMpiTest(exchange_keys.cpp exchange_keys GlobalKeyExchange 2)
addMpiTest(focus_transfer.cpp focus_transfer FocusTransfer 2)
addMpiTest(domain_2ranks.cpp domain_2ranks GlobalDomain2Ranks 2)

addMpiTest(treedomain.cpp treedomain GlobalDomainTreeIntregration 5)
addMpiTest(exchange_general.cpp exchange_general GeneralFocusExchange 5)
addMpiTest(exchange_domain.cpp exchange_domain GlobalDomainExchange 5)
addMpiTest(focus_tree.cpp focus_tree FocusTreeIntregration 5)
addMpiTest(domain_nranks.cpp domain_nranks GlobalDomainNRanks 5)

if (CMAKE_CUDA_COMPILER)
    # workaround for compiling MPI with nvcc on cray systems
    if (MPI_CXX_INCLUDE_PATH STREQUAL "" AND DEFINED ENV{CRAY_MPICH2_DIR})
        message(STATUS "Applying MPI include workaround on CRAY")
        set(MPI_CXX_INCLUDE_PATH $ENV{MPICH_DIR}/include)
    endif()

    addMpiTest(exchange_halos_gpu.cpp exchange_halos_gpu GlobalHaloExchangeGpu 2)
    target_link_libraries(exchange_halos_gpu CUDA::cudart)
endif()

