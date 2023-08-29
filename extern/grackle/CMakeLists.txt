cmake_minimum_required(VERSION 3.17)
project(grackle C Fortran)

find_package(HDF5 REQUIRED)

set(grackle_SRCS
        grackle_deps/auto_show_config.c
        grackle_deps/auto_show_flags.c
        grackle_deps/auto_show_version.c
        grackle_repo/src/clib/calculate_cooling_time.c
        grackle_repo/src/clib/calculate_dust_temperature.c
        grackle_repo/src/clib/calculate_gamma.c
        grackle_repo/src/clib/calculate_pressure.c
        grackle_repo/src/clib/calculate_temperature.c
        grackle_repo/src/clib/grackle_units.c
        grackle_repo/src/clib/index_helper.c
        grackle_repo/src/clib/initialize_chemistry_data.c
        grackle_repo/src/clib/initialize_cloudy_data.c
        grackle_repo/src/clib/initialize_UVbackground_data.c
        grackle_repo/src/clib/set_default_chemistry_parameters.c
        grackle_repo/src/clib/solve_chemistry.c
        grackle_repo/src/clib/update_UVbackground_rates.c
        grackle_repo/src/clib/rate_functions.c
        grackle_repo/src/clib/initialize_rates.c
        grackle_repo/src/clib/calc_temp1d_cloudy_g.F
        grackle_repo/src/clib/calc_temp_cloudy_g.F
        grackle_repo/src/clib/calc_tdust_1d_g.F
        grackle_repo/src/clib/calc_tdust_3d_g.F
        grackle_repo/src/clib/cool1d_cloudy_g.F
        grackle_repo/src/clib/cool1d_cloudy_old_tables_g.F
        grackle_repo/src/clib/cool1d_multi_g.F
        grackle_repo/src/clib/cool_multi_time_g.F
        grackle_repo/src/clib/interpolators_g.F
        grackle_repo/src/clib/solve_rate_cool_g.F
        )

add_library(grackle ${grackle_SRCS})

set(PRECISION_DEFINE GRACKLE_FLOAT_8)
configure_file(${PROJECT_SOURCE_DIR}/config/grackle_float.h.in ${CMAKE_BINARY_DIR}/extern/grackle/grackle_deps/grackle_float.h)

target_include_directories(grackle PUBLIC ${PROJECT_SOURCE_DIR}/grackle_repo/src/clib)
target_include_directories(grackle PUBLIC ${PROJECT_BINARY_DIR}/grackle_deps)
target_include_directories(grackle PRIVATE ${HDF5_INCLUDE_DIRS} ${MPI_C_INCLUDE_PATH})
target_link_libraries(grackle ${HDF5_LIBRARIES} ${MPI_C_LIBRARIES} gfortran)
#Omit warning of deprecated functions (already from removed SPH-EXA)
target_compile_definitions(grackle PUBLIC OMIT_LEGACY_INTERNAL_GRACKLE_FUNC)
target_compile_definitions(grackle PRIVATE LINUX H5_USE_16_API PIC)
set_target_properties(grackle PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Language specific compilation flags.
target_compile_options(grackle PRIVATE
        $<$<COMPILE_LANGUAGE:Fortran>:-fno-second-underscore -ffixed-line-length-132>
        )

configure_file(grackle_deps/version.h.in ${CMAKE_BINARY_DIR}/extern/grackle/grackle_deps/version.h)