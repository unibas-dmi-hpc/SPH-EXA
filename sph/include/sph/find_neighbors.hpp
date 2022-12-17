#pragma once

#include "cstone/findneighbors.hpp"

namespace sph
{

template<class T, class KeyType, class Dataset>
void findNeighborsSfc(size_t startIndex, size_t endIndex, unsigned ngmax, Dataset& d,
                      cstone::OctreeNsView<T, KeyType> treeView, const cstone::Box<T>& box)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{}) { return; }

    cstone::findNeighborsT(d.x.data(), d.y.data(), d.z.data(), d.h.data(), startIndex, endIndex, box, treeView, ngmax,
                           d.neighbors.data(), d.nc.data() + startIndex);
}

} // namespace sph
