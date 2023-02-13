#pragma once

#include "cstone/findneighbors.hpp"

namespace sph
{

using cstone::LocalIndex;

template<class T, class KeyType>
void findNeighborsSph(const T* x, const T* y, const T* z, T* h, LocalIndex firstId, LocalIndex lastId,
                      const cstone::Box<T>& box, const cstone::OctreeNsView<T, KeyType>& treeView, unsigned ng0,
                      unsigned ngmax, LocalIndex* neighbors, unsigned* nc)
{
    LocalIndex numWork = lastId - firstId;

    unsigned ngmin = ng0 / 4;

    size_t        numFails     = 0;
    constexpr int maxIteration = 10;

#pragma omp parallel for reduction(+ : numFails)
    for (LocalIndex i = 0; i < numWork; ++i)
    {
        LocalIndex id = i + firstId;
        nc[i]         = findNeighbors(id, x, y, z, h, treeView, box, ngmax, neighbors + i * ngmax);

        int iteration = 0;
        while (ngmin > nc[i] || nc[i] > ngmax && iteration++ < maxIteration)
        {
            h[id] = updateH(ng0, nc[i], h[id]);
            nc[i] = findNeighbors(id, x, y, z, h, treeView, box, ngmax, neighbors + i * ngmax);
        }
        numFails += (iteration == maxIteration);
    }

    if (numFails)
    {
        std::cout << "Coupled h-neighbor count updated failed to converge for " << numFails << " particles"
                  << std::endl;
    }
}

//! @brief perform neighbor search together with updating the smoothing lengths
template<class T, class Dataset>
void findNeighborsSfc(size_t startIndex, size_t endIndex, Dataset& d, const cstone::Box<T>& box)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{}) { return; }

    if (d.ng0 > d.ngmax) { throw std::runtime_error("ng0 should be smaller than ngmax\n"); }

    findNeighborsSph(d.x.data(), d.y.data(), d.z.data(), d.h.data(), startIndex, endIndex, box, d.treeView, d.ng0,
                     d.ngmax, d.neighbors.data(), d.nc.data() + startIndex);
}

} // namespace sph
