#pragma once

#include "cstone/findneighbors.hpp"

namespace sph
{

template<class T, class Dataset>
void findNeighborsSfc(size_t startIndex, size_t endIndex, unsigned ngmax, Dataset& d, const cstone::Box<T>& box)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{}) { return; }

    cstone::findNeighbors(d.x.data(), d.y.data(), d.z.data(), d.h.data(), startIndex, endIndex, box, d.treeView, ngmax,
                          d.neighbors.data(), d.nc.data() + startIndex);
}

} // namespace sph
