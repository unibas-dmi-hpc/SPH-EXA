
add_library(cudaFindNeighbors OBJECT findneighbors.cu)
target_include_directories(cudaFindNeighbors PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_library(gatherObj OBJECT gather.cu)