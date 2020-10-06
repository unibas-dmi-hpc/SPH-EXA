#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>
#include <tuple>
#include <vector>

#include "zorder.hpp"

namespace sphexa
{



template<class T>
unsigned treeLevel(T radius, T xmin, T xmax, T ymin, T ymax, T zmin, T zmax)
{
    std::array<T, 3> boxRanges{ xmax-xmin, ymax-ymin, zmax-zmin };

    T maxRange = *std::max_element(begin(boxRanges), end(boxRanges));
    T radiusNormalized = radius / maxRange;

    return unsigned(-log2(radiusNormalized));
}

template<class T, class I>
void findNeighbors(int id, const T* x, const T* y, const T* z, unsigned treeLevel,
                   const I* mortonCodes, int *neighbors, int *neighborsCount,
                   int ngmax)
{

}

} // namespace sphexa
