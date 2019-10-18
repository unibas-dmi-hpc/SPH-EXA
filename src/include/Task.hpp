#pragma once

#include <vector>

namespace sphexa
{
struct Task
{
    Task(int size) : clist(size), neighbors(size * ngmax), neighborsCount(size) {}

    std::vector<int> clist;
    std::vector<int> neighbors;
    std::vector<int> neighborsCount;

    constexpr static size_t ngmax = 650;
};
} // namespace sphexa
