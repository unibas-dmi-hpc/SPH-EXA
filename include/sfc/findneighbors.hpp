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
unsigned treeLevel(T radius, T maxRange)
{
    T radiusNormalized = radius / maxRange;
    return unsigned(-log2(radiusNormalized));
}

template<class T, class I>
void findNeighbors(int id, const T* x, const T* y, const T* z, T radius, T boxMaxRange,
                   const I* mortonCodes, int *neighbors, int *neighborsCount,
                   int n, int ngmax)
{
    T radiusSq = radius * radius;
    unsigned depth = treeLevel(radius, boxMaxRange);
    I mortonCode = mortonCodes[id];

    std::array<I, 27> neighborCodes;

    // find neighboring boxes / octree nodes
    int nBoxes = 0;
    for (int dx = -1; dx < 2; ++dx)
        for (int dy = -1; dy < 2; ++dy)
            for (int dz = -1; dz < 2; ++dz)
                neighborCodes[nBoxes++] = mortonNeighbor(mortonCode, depth, dx, dy, dz);

    std::sort(begin(neighborCodes), begin(neighborCodes) + nBoxes);
    auto last = std::unique(begin(neighborCodes), begin(neighborCodes) + nBoxes);

    T xi = x[id], yi = y[id], zi = z[id];

    int ngcount = 0;
    for (auto neighbor = begin(neighborCodes); neighbor != last; ++neighbor)
    {
        int startIndex = std::lower_bound(mortonCodes, mortonCodes + n, *neighbor) - mortonCodes;
        int endIndex   = std::upper_bound(mortonCodes, mortonCodes + n, *neighbor + nodeRange<I>(depth)) - mortonCodes;

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
