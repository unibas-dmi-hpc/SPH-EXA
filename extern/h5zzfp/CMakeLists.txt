#------------------------------------------------------------------------------#
# ZFP
#------------------------------------------------------------------------------#
set(ZFP_SOURCE_DIR "zfp/src")
set(ZFP_INCLUDE_DIR "zfp/include")
set(zfp_source
${ZFP_SOURCE_DIR}/zfp.c
${ZFP_SOURCE_DIR}/bitstream.c
${ZFP_SOURCE_DIR}/traitsf.h ${ZFP_SOURCE_DIR}/traitsd.h ${ZFP_SOURCE_DIR}/block1.h ${ZFP_SOURCE_DIR}/block2.h ${ZFP_SOURCE_DIR}/block3.h ${ZFP_SOURCE_DIR}/block4.h
${ZFP_SOURCE_DIR}/encode1f.c ${ZFP_SOURCE_DIR}/encode1d.c ${ZFP_SOURCE_DIR}/encode1i.c ${ZFP_SOURCE_DIR}/encode1l.c
${ZFP_SOURCE_DIR}/decode1f.c ${ZFP_SOURCE_DIR}/decode1d.c ${ZFP_SOURCE_DIR}/decode1i.c ${ZFP_SOURCE_DIR}/decode1l.c
${ZFP_SOURCE_DIR}/encode2f.c ${ZFP_SOURCE_DIR}/encode2d.c ${ZFP_SOURCE_DIR}/encode2i.c ${ZFP_SOURCE_DIR}/encode2l.c
${ZFP_SOURCE_DIR}/decode2f.c ${ZFP_SOURCE_DIR}/decode2d.c ${ZFP_SOURCE_DIR}/decode2i.c ${ZFP_SOURCE_DIR}/decode2l.c
${ZFP_SOURCE_DIR}/encode3f.c ${ZFP_SOURCE_DIR}/encode3d.c ${ZFP_SOURCE_DIR}/encode3i.c ${ZFP_SOURCE_DIR}/encode3l.c
${ZFP_SOURCE_DIR}/decode3f.c ${ZFP_SOURCE_DIR}/decode3d.c ${ZFP_SOURCE_DIR}/decode3i.c ${ZFP_SOURCE_DIR}/decode3l.c
${ZFP_SOURCE_DIR}/encode4f.c ${ZFP_SOURCE_DIR}/encode4d.c ${ZFP_SOURCE_DIR}/encode4i.c ${ZFP_SOURCE_DIR}/encode4l.c
${ZFP_SOURCE_DIR}/decode4f.c ${ZFP_SOURCE_DIR}/decode4d.c ${ZFP_SOURCE_DIR}/decode4i.c ${ZFP_SOURCE_DIR}/decode4l.c)

add_library(zfp ${zfp_source})
target_link_libraries(zfp PRIVATE OpenMP::OpenMP_C)
target_compile_definitions(zfp
  PRIVATE ${zfp_private_defs}
  PUBLIC ${zfp_public_defs}
)
target_include_directories(zfp
  PUBLIC
    ${ZFP_INCLUDE_DIR}
)

#------------------------------------------------------------------------------#
# H5Z-ZFP
#------------------------------------------------------------------------------#
set(H5Z_ZFP_HEADERS
  H5Zzfp.h H5Zzfp_lib.h H5Zzfp_plugin.h H5Zzfp_props.h H5Zzfp_version.h)

set(H5Z_ZFP_LIB_SOURCE
  ${H5Z_ZFP_HEADERS}
  H5Zzfp_props_private.h
  H5Zzfp.c H5Zzfp_props.c)

if ((HDF5_IS_PARALLEL OR HDF5_ENABLE_PARALLEL) AND ("${HDF5_VERSION}" VERSION_EQUAL "1.14.0"))
  find_package(MPI REQUIRED COMPONENTS C)
endif()

add_library(h5z_zfp STATIC ${H5Z_ZFP_LIB_SOURCE})
target_compile_definitions(h5z_zfp PRIVATE H5Z_ZFP_AS_LIB)
target_include_directories(h5z_zfp
  PUBLIC
    ${ZFP_INCLUDE_DIR}
    ${HDF5_INCLUDE_DIRS}
)
target_link_libraries(h5z_zfp PUBLIC zfp ${HDF5_LIBRARIES})
set_target_properties(h5z_zfp PROPERTIES POSITION_INDEPENDENT_CODE 1)