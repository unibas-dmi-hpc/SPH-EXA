include(sphexa_add_test)

function(addMpiTest source exename testname ranks)
    add_executable(${exename} ${source} test_main.cpp)
    target_include_directories(${exename} PRIVATE ../../include)
    target_include_directories(${exename} PRIVATE ../)
    target_include_directories(${exename} PRIVATE ${MPI_CXX_INCLUDE_PATH})
    target_compile_options(${exename} PRIVATE -Wno-unknown-pragmas)
    target_link_libraries(${exename} ${MPI_CXX_LIBRARIES} GTest::gtest_main)
    sphexa_add_test(${testname} EXECUTABLE ${exename} RANKS ${ranks})
    install(TARGETS ${exename} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/integration_mpi)
endfunction()

addMpiTest(globaloctree.cpp globaloctree GlobalTreeTests 2)

addMpiTest(exchange_domain.cpp exchange_domain GlobalDomainExchange 2)

addMpiTest(treedomain.cpp treedomain GlobalDomainTreeIntregration 2)

addMpiTest(focus_tree.cpp focus_tree FocusTreeIntregration 2)

addMpiTest(exchange_halos.cpp exchange_halos GlobalHaloExchange 2)

addMpiTest(domain_nranks.cpp domain_nranks GlobalDomainNRanks 2)

addMpiTest(box_mpi.cpp box_mpi GlobalBox 2)

addMpiTest(exchange_focus.cpp exchange_focus GlobalFocusExchange 2)

addMpiTest(exchange_general.cpp exchange_general GeneralFocusExchange 2)

addMpiTest(exchange_keys.cpp exchange_keys GlobalKeyExchange 2)

addMpiTest(focus_transfer.cpp focus_transfer FocusTransfer 2)

addMpiTest(domain_2ranks.cpp domain_2ranks GlobalDomain2Ranks 2)
