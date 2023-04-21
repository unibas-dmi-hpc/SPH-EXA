
add_library(cosmo cosmology_data.hpp utils.hpp static.hpp cdm.hpp lcdm.hpp factory.hpp)
set_target_properties(cosmo PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(cosmo PRIVATE ${CSTONE_DIR} ${COSMO_DIR})
target_compile_definitions(cosmo PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
