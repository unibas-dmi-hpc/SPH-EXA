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
static inline T distancesq(const T x1, const T y1, const T z1, const T x2, const T y2, const T z2)
{
    T xx = x1 - x2;
    T yy = y1 - y2;
    T zz = z1 - z2;

    return xx * xx + yy * yy + zz * zz;
}

template<class T>
unsigned treeLevel(T radius, T xmin, T xmax, T ymin, T ymax, T zmin, T zmax)
{
    std::array<T, 3> boxRanges{ xmax-xmin, ymax-ymin, zmax-zmin };

    T maxRange = *std::max_element(begin(boxRanges), end(boxRanges));
    T radiusNormalized = radius / maxRange;

    return unsigned(-log2(radiusNormalized));
}

template<class T, class I>
void findNeighbors(int id, const T* x, const T* y, const T* z, T radius, std::array<T, 6> box,
                   const I* mortonCodes, int *neighbors, int *neighborsCount,
                   int n, int ngmax)
{
    T radiusSq = radius * radius;
    unsigned depth = treeLevel(radius, box[0], box[1], box[2], box[3], box[4], box[5]);
    I currentCode = mortonCodes[id];
    I homeBox     = detail::enclosingBoxCode(currentCode, depth);

    std::vector<std::tuple<int, int>> ranges;
    ranges.reserve(27);

    // find neighboring boxes / octree nodes
    for (int dx = -1; dx < 2; ++dx)
    {
        for (int dy = -1; dy < 2; ++dy)
        {
            for (int dz = -1; dz < 2; ++dz)
            {
                I neighborStart = mortonNeighbor(homeBox, depth, dx, dy, dz);
                I neighborEnd   = neighborStart + nodeRange<I>(depth);

                auto itStart = std::lower_bound(mortonCodes, mortonCodes + n, neighborStart);
                auto itEnd   = std::upper_bound(mortonCodes, mortonCodes + n, neighborEnd);

                int startIndex = std::distance(mortonCodes, itStart);
                int endIndex   = std::distance(mortonCodes, itEnd);

                ranges.emplace_back(startIndex, endIndex);
            }
        }
    }

    std::sort(begin(ranges), end(ranges));
    auto last = std::unique(begin(ranges), end(ranges));
    ranges.erase(last, end(ranges));

    T xi = x[id], yi = y[id], zi = z[id];
    int ngcount = 0;

    for (auto range : ranges)
    {
        auto startIndex = std::get<0>(range);
        auto endIndex = std::get<1>(range);

        for (int j = startIndex; j < endIndex; ++j)
        {
            if (j == id) { continue; }

            if (distancesq(xi, yi, zi, x[j], y[j], z[j]) < radiusSq)
            {
                neighbors[id * ngmax + ngcount++] = j;
            }

            if (ngcount == ngmax) { break; }
        }
    }

    neighborsCount[id] = ngcount;
}

} // namespace sphexa
