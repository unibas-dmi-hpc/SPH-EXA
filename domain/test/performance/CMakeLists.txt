
add_executable(octree_perf octree.cpp)

target_include_directories(octree_perf PRIVATE ../../include)
target_include_directories(octree_perf PRIVATE ../)
target_link_libraries(octree_perf PUBLIC OpenMP::OpenMP_CXX)