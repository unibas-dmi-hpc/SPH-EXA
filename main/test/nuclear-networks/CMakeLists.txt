set(nnet hydro-mockup)

add_executable(${nnet} hydro-mockup.cpp)

target_include_directories(${nnet} PRIVATE ${SPH_DIR})
target_include_directories(${nnet} PRIVATE ${CSTONE_DIR})
target_include_directories(${nnet} PRIVATE ${SPHNNET_DIR})
target_include_directories(${nnet} PRIVATE ${NNET_DIR})

target_include_directories(${nnet} PUBLIC ${MPI_CXX_INCLUDE_DIRS})

target_compile_definitions(${nnet} PRIVATE USE_MPI)
target_link_libraries(${nnet} PRIVATE ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX
	nuclear)

install(TARGETS ${nnet} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})





option(CUDA_HYDRO_MOCKUP "Enable CUDA hydro mockut test for nuclear networks cooling" OFF)

if(CUDA_HYDRO_MOCKUP)
	if(CMAKE_HIP_COMPILER)
		set_source_properties(hydro-mockup.cpp PROPERTY LANGUAGE HIP)
	endif()

	if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
	    add_executable(${nnet}-cuda hydro-mockup.cpp)
	  # add_executable(${nnet}-cuda hydro-mockup.cpp)

	   	set_target_properties(${nnet}-cuda PROPERTIES
	   		CUDA_SEPARABLE_COMPILATION ON
	    )

		target_include_directories(${nnet}-cuda PRIVATE ${SPH_DIR})
		target_include_directories(${nnet}-cuda PRIVATE ${CSTONE_DIR})
		target_include_directories(${nnet}-cuda PRIVATE ${SPHNNET_DIR})
		target_include_directories(${nnet}-cuda PRIVATE ${NNET_DIR})

		target_include_directories(${nnet}-cuda PUBLIC ${MPI_CXX_INCLUDE_DIRS})

		target_compile_definitions(${nnet}-cuda PRIVATE USE_MPI USE_CUDA)
		target_link_libraries(${nnet}-cuda PRIVATE OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES} CUDA::cudart
			nuclear_gpu reallocate_obj)

		install(TARGETS ${nnet}-cuda RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})






	    add_executable(${nnet}-cpu hydro-mockup.cpp)
	  # add_executable(${nnet}-cpu hydro-mockup.cpp)

	   	set_target_properties(${nnet}-cpu PROPERTIES
	   		CUDA_SEPARABLE_COMPILATION ON
	    )

		target_include_directories(${nnet}-cpu PRIVATE ${SPH_DIR})
		target_include_directories(${nnet}-cpu PRIVATE ${CSTONE_DIR})
		target_include_directories(${nnet}-cpu PRIVATE ${SPHNNET_DIR})
		target_include_directories(${nnet}-cpu PRIVATE ${NNET_DIR})

		target_include_directories(${nnet}-cpu PUBLIC ${MPI_CXX_INCLUDE_DIRS})

		target_compile_definitions(${nnet}-cpu PRIVATE USE_MPI USE_CUDA CUDA_CPU_TEST)
		target_link_libraries(${nnet}-cpu PRIVATE OpenMP::OpenMP_CXX ${MPI_CXX_LIBRARIES} CUDA::cudart
			nuclear_gpu reallocate_obj)

		install(TARGETS ${nnet}-cpu RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
	endif()
endif()
