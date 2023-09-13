project(H5Part C)

#--------------------------------------------------------
# dependencies
#--------------------------------------------------------
find_package(HDF5 REQUIRED)

#--------------------------------------------------------
# define empty library (add settings later)
#--------------------------------------------------------
add_library(H5Part STATIC)

#--------------------------------------------------------
# Source files
#--------------------------------------------------------
set(H5Part_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/H5Part.c
  ${CMAKE_CURRENT_SOURCE_DIR}/h5tools_type.c
)
target_sources(H5Part PRIVATE ${H5Part_SRCS})

#--------------------------------------------------------
# Windows: Make sure msvc builds as cxx not C
# This resolves some struct typedef issues
#--------------------------------------------------------
if(WIN32)
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/H5Part.c
    #${CMAKE_CURRENT_SOURCE_DIR}/H5Block.c
    PROPERTIES LANGUAGE CXX
  )

  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

#--------------------------------------------------------
# Parallel compilation settings
#--------------------------------------------------------
if(HDF5_IS_PARALLEL)
  find_package(MPI REQUIRED)
  target_compile_definitions(H5Part PUBLIC H5PART_PARALLEL_IO)
  target_link_libraries(H5Part ${MPI_C_LIBRARIES})
  target_include_directories(H5Part PRIVATE ${MPI_C_INCLUDE_DIRS})
else()
  message("HDF5_IS_PARALLEL not found, Parallel HDF5 IO will not be enabled")
endif()

#--------------------------------------------------------
# libraries
#--------------------------------------------------------
target_link_libraries(H5Part ${HDF5_LIBRARIES})

#--------------------------------------------------------
# Includes, configuration headers
#--------------------------------------------------------
target_include_directories(H5Part PRIVATE ${PROJECT_SOURCE_DIR})
target_include_directories(H5Part PUBLIC ${HDF5_INCLUDE_DIRS})

#--------------------------------------------------------
# API compatibility setting
#--------------------------------------------------------
target_compile_definitions(H5Part PUBLIC H5_USE_110_API)

#--------------------------------------------------------
# Make sure static lib can be linked into dynamic one
#--------------------------------------------------------
if(BUILD_SHARED_LIBS AND CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_COMPILER_IS_GNUCC)
  set_target_properties(H5Part PROPERTIES COMPILE_FLAGS -fPIC)
endif()
