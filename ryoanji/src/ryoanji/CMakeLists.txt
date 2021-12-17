
add_library(cstone_tree treebuilder.cu)
target_include_directories(cstone_tree PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_include_directories(cstone_tree PUBLIC ${CMAKE_SOURCE_DIR}/domain/include)
