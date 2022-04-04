
function(enableH5Part exename)
    if (SPH_EXA_WITH_H5PART)
        target_compile_definitions(${exename} PUBLIC SPH_EXA_HAVE_H5PART)
        target_include_directories(${exename} PUBLIC ${PROJECT_SOURCE_DIR}/extern/h5part)
        target_link_libraries(${exename} PRIVATE H5Part ${HDF5_LIBRARIES})
    endif()
endfunction()

find_package(Git)
if(Git_FOUND)
    if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
  execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  execute_process(
    COMMAND git log -1 --format=%h
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
else(EXISTS "${CMAKE_SOURCE_DIR}/.git")
  set(GIT_BRANCH "-")
  set(GIT_COMMIT_HASH "-")
endif(EXISTS "${CMAKE_SOURCE_DIR}/.git")
endif(Git_FOUND)
configure_file(
  version.h.in
  ${CMAKE_BINARY_DIR}/main/src/sphexa/version.h
)

set(exename sphexa)
add_executable(${exename} sphexa.cpp)
target_include_directories(${exename} PRIVATE ${SPH_EXA_INCLUDE_DIRS})
target_include_directories(${exename} PRIVATE ${CMAKE_BINARY_DIR}/main/src/sphexa)
target_link_libraries(${exename} PRIVATE ${MPI_CXX_LIBRARIES})
target_link_libraries(${exename} PRIVATE OpenMP::OpenMP_CXX)
enableH5Part(${exename})
install(TARGETS ${exename} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(INSITU STREQUAL "Catalyst")
    target_include_directories(${exename} PRIVATE ..)
    target_compile_definitions(${exename}
	    PRIVATE
	    "PARAVIEW_IMPL_DIR=\"${ParaView_CATALYST_DIR}\""
	    SPH_EXA_USE_CATALYST2)
    target_link_libraries(${exename} PRIVATE catalyst::catalyst)
elseif(INSITU STREQUAL "Ascent")
    target_include_directories(${exename} PRIVATE ..)
    target_compile_definitions(${exename}
	    PRIVATE
	    SPH_EXA_USE_ASCENT)
    target_link_libraries(${exename} PRIVATE ascent::ascent_mpi)
endif()

if(CMAKE_CUDA_COMPILER)
    add_executable(${exename}-cuda $<TARGET_OBJECTS:gather_obj> $<TARGET_OBJECTS:cuda_find_neighbors_obj> $<TARGET_OBJECTS:cuda_sph> sphexa.cpp)
    target_include_directories(${exename}-cuda PRIVATE ${SPH_EXA_INCLUDE_DIRS})
    target_include_directories(${exename}-cuda PUBLIC ${CMAKE_BINARY_DIR}/main/src/sphexa)
    target_compile_definitions(${exename}-cuda PRIVATE USE_MPI USE_CUDA)
    set_target_properties(${exename}-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(${exename}-cuda PRIVATE ${MPI_CXX_LIBRARIES})
    target_link_libraries(${exename}-cuda PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(${exename}-cuda PRIVATE CUDA::cudart)
    enableH5Part(${exename}-cuda)
    install(TARGETS ${exename}-cuda RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

elseif(CMAKE_HIP_COMPILER)
    set_source_files_properties(sphexa.cpp PROPERTIES LANGUAGE HIP)
    add_executable(${exename}-hip $<TARGET_OBJECTS:gather_obj> $<TARGET_OBJECTS:cuda_find_neighbors_obj> $<TARGET_OBJECTS:cuda_sph> sphexa.cpp)
    set_property(TARGET ${exename}-hip PROPERTY HIP_ARCHITECTURES gfx908)
    target_include_directories(${exename}-hip PRIVATE ${SPH_EXA_INCLUDE_DIRS})
    target_include_directories(${exename}-hip PUBLIC ${CMAKE_BINARY_DIR}/main/src/sphexa)
    target_include_directories(${exename}-hip PRIVATE ${HIP_PATH}/include)
    target_compile_definitions(${exename}-hip PRIVATE USE_MPI USE_CUDA)
    target_link_libraries(${exename}-hip PRIVATE ${MPI_CXX_LIBRARIES})
    target_link_libraries(${exename}-hip PRIVATE OpenMP::OpenMP_CXX)
    enableH5Part(${exename}-hip)
    install(TARGETS sedov-hip RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
