#cmake_minimum_required(VERSION 3.17)
#project(grackle C Fortran)

#find_package(HDF5 REQUIRED)
#
#set(grackle_SRCS
#    ${CMAKE_CURRENT_SOURCE_DIR}/grackle_deps/src/clib/auto_show_config.c
#    ${CMAKE_CURRENT_SOURCE_DIR}/grackle_deps/src/clib/auto_show_flags.c
#    ${CMAKE_CURRENT_SOURCE_DIR}/grackle_deps/src/clib/auto_show_version.c
#    ${CMAKE_CURRENT_SOURCE_DIR}/grackle/src/clib/calculate_cooling_time.c
#    ${CMAKE_CURRENT_SOURCE_DIR}/grackle/src/clib/calculate_dust_temperature.c
#    ${CMAKE_CURRENT_SOURCE_DIR}/grackle/src/clib/calculate_gamma.c
#    ${CMAKE_CURRENT_SOURCE_DIR}/grackle/src/clib/calculate_pressure.c
#    ${CMAKE_CURRENT_SOURCE_DIR}/grackle/src/clib/calculate_temperature.c
#    ${CMAKE_CURRENT_SOURCE_DIR}/grackle/src/clib/grackle_units.c
#    ${CMAKE_CURRENT_SOURCE_DIR}/grackle/src/clib/initialize_chemistry_data.c
#    ${CMAKE_CURRENT_SOURCE_DIR}/grackle/src/clib/initialize_cloudy_data.c
#    ${CMAKE_CURRENT_SOURCE_DIR}/grackle/src/clib/initialize_UVbackground_data.c
#    ${CMAKE_CURRENT_SOURCE_DIR}/grackle/src/clib/set_default_chemistry_parameters.c
#    ${CMAKE_CURRENT_SOURCE_DIR}/grackle/src/clib/solve_chemistry.c
#    ${CMAKE_CURRENT_SOURCE_DIR}/grackle/src/clib/update_UVbackground_rates.c
#    ${CMAKE_CURRENT_SOURCE_DIR}/grackle/src/clib/rate_functions.c
#    ${CMAKE_CURRENT_SOURCE_DIR}/grackle/src/clib/initialize_rates.c
#    ${CMAKE_CURRENT_SOURCE_DIR}/grackle/src/clib/calc_temp1d_cloudy_g.F
#    ${CMAKE_CURRENT_SOURCE_DIR}/grackle/src/clib/calc_temp_cloudy_g.F
#    ${CMAKE_CURRENT_SOURCE_DIR}/grackle/src/clib/calc_tdust_1d_g.F
#    ${CMAKE_CURRENT_SOURCE_DIR}/grackle/src/clib/calc_tdust_3d_g.F
#    ${CMAKE_CURRENT_SOURCE_DIR}/grackle/src/clib/cool1d_cloudy_g.F
#    ${CMAKE_CURRENT_SOURCE_DIR}/grackle/src/clib/cool1d_cloudy_old_tables_g.F
#    ${CMAKE_CURRENT_SOURCE_DIR}/grackle/src/clib/cool1d_multi_g.F
#    ${CMAKE_CURRENT_SOURCE_DIR}/grackle/src/clib/cool_multi_time_g.F
#    ${CMAKE_CURRENT_SOURCE_DIR}/grackle/src/clib/interpolators_g.F
#    ${CMAKE_CURRENT_SOURCE_DIR}/grackle/src/clib/solve_rate_cool_g.F
#)
#
#link_directories(${CMAKE_SOURCE_DIR}/extern/grackle)
#set(CMAKE_Fortran_FLAGS "-fno-second-underscore -ffixed-line-length-132")
#add_library(grackle SHARED ${grackle_SRCS})
#target_include_directories(grackle PUBLIC ${HDF5_INCLUDE_DIRS})
#target_link_libraries(grackle ${HDF5_LIBRARIES})
#target_include_directories(grackle PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/grackle/src/clib)
#target_compile_definitions(grackle PRIVATE LINUX H5_USE_16_API CONFIG_BFLOAT_8 PIC)
#set_target_properties(grackle PROPERTIES COMPILE_FLAGS -fPIC)
# TODO: install
