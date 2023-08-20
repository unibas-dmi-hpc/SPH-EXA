# Define headers.
set(h5z_zfp_headers
  H5Zzfp.h H5Zzfp_lib.h H5Zzfp_plugin.h H5Zzfp_props.h H5Zzfp_version.h)

#------------------------------------------------------------------------------#
# Static library
#------------------------------------------------------------------------------#
# Define source
set(h5z_zfp_lib_source
  ${h5z_zfp_headers}
  H5Zzfp_props_private.h
  H5Zzfp.c H5Zzfp_props.c)

if ((HDF5_IS_PARALLEL OR HDF5_ENABLE_PARALLEL) AND ("${HDF5_VERSION}" VERSION_EQUAL "1.14.0"))
  find_package(MPI REQUIRED COMPONENTS C)
endif()

set(ZFP_HOME "/home/appcell/spack/opt/spack/linux-ubuntu22.04-skylake/gcc-11.3.0/zfp-1.0.0-xvspbost7f5xtympkn442j3nzqcdghzj/")
set(ZFP_CMAKE_PATH "${ZFP_HOME}/lib/cmake/zfp")
set(ZFP_LIB "${ZFP_HOME}/lib/libzfp.so")
set(ZFP_INCLUDE_DIR "${ZFP_HOME}/include")

find_package(ZFP REQUIRED CONFIG PATHS ${ZFP_CMAKE_PATH})
if (NOT ZFP_FOUND)
  message (FATAL_ERROR "ZFP is Required for h5z_zfp")
endif ()
add_library(h5z_zfp_static STATIC ${h5z_zfp_lib_source})
target_compile_definitions(h5z_zfp_static PRIVATE H5Z_ZFP_AS_LIB)
target_include_directories(h5z_zfp_static
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    ${ZFP_INCLUDE_DIR}
    ${HDF5_INCLUDE_DIRS}
)
target_link_libraries(h5z_zfp_static PUBLIC ${ZFP_LIB} ${HDF5_LIBRARIES})
set_target_properties(h5z_zfp_static PROPERTIES POSITION_INDEPENDENT_CODE 1)