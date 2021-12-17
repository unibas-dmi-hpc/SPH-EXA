
function(addMpiTest source exename testname)
    add_executable(${exename} ${source} test_main.cpp)
    target_include_directories(${exename} PRIVATE ../../include)
    target_include_directories(${exename} PRIVATE ../)
    target_include_directories(${exename} PRIVATE ${MPI_CXX_INCLUDE_PATH})
    target_compile_options(${exename} PRIVATE -Wno-unknown-pragmas)
    target_link_libraries(${exename} ${MPI_CXX_LIBRARIES} gtest_main)
    add_test(NAME ${testname} COMMAND ${exename})
    install(TARGETS ${exename} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/integration_mpi)
endfunction()

addMpiTest(globaloctree.cpp globaloctree GlobalTreeTests)

addMpiTest(exchange_domain.cpp exchange_domain GlobalDomainExchange)

addMpiTest(treedomain.cpp treedomain GlobalDomainTreeIntregration)

addMpiTest(focus_tree.cpp focus_tree FocusTreeIntregration)

addMpiTest(exchange_halos.cpp exchange_halos GlobalHaloExchange)

addMpiTest(domain_nranks.cpp domain_nranks GlobalDomainNRanks)

addMpiTest(box_mpi.cpp box_mpi GlobalBox)

addMpiTest(exchange_focus.cpp exchange_focus GlobalFocusExchange)

addMpiTest(exchange_keys.cpp exchange_keys GlobalKeyExchange)

addMpiTest(domain_2ranks.cpp domain_2ranks GlobalDomain2Ranks)
