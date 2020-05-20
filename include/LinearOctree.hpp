#pragma once

#include <vector>
#include "Octree.hpp"

namespace sphexa
{
template <typename T>
struct LinearOctree
{
    int size;
    std::vector<int> ncells;
    std::vector<int> cells;
    std::vector<int> localPadding;
    std::vector<int> localParticleCount;
    std::vector<T> xmin, xmax, ymin, ymax, zmin, zmax;
};

template <typename T>
size_t getNumberOfNodesRec(const Octree<T> &o)
{
    size_t count = 1;
    if ((int)o.cells.size() == o.ncells)
        for (int i = 0; i < o.ncells; i++)
            count += getNumberOfNodesRec(*o.cells[i]);
    return count;
}

template <typename T>
size_t createLinearOctreeRec(const Octree<T> &o, LinearOctree<T> &l, size_t it = 0)
{
    l.localPadding[it] = o.localPadding;
    l.ncells[it] = o.cells.size();
    l.localParticleCount[it] = o.localParticleCount;
    l.xmin[it] = o.xmin;
    l.xmax[it] = o.xmax;
    l.ymin[it] = o.ymin;
    l.ymax[it] = o.ymax;
    l.zmin[it] = o.zmin;
    l.zmax[it] = o.zmax;

    int count = 1;

    if ((int)o.cells.size() == o.ncells)
    {
        for (int i = 0; i < o.ncells; i++)
        {
            l.cells[it * 8 + i] = it + count;
            count += createLinearOctreeRec(*o.cells[i], l, it + count);
        }
    }

    return count;
}

template <typename T>
void createLinearOctree(const Octree<T> &o, LinearOctree<T> &l)
{
    size_t nodeCount = getNumberOfNodesRec(o);

    l.size = nodeCount;
    l.ncells.resize(nodeCount);
    l.cells.resize(8 * nodeCount);
    l.localPadding.resize(nodeCount);
    l.localParticleCount.resize(nodeCount);
    l.xmin.resize(nodeCount);
    l.xmax.resize(nodeCount);
    l.ymin.resize(nodeCount);
    l.ymax.resize(nodeCount);
    l.zmin.resize(nodeCount);
    l.zmax.resize(nodeCount);

    createLinearOctreeRec(o, l);
}

} // namespace sphexa
