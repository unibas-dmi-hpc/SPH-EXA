/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! \file
 * \brief  Find neighbors in Morton code sorted x,y,z arrays
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */


#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

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

/*! \brief determines the octree subdivision layer at which the node size in
 *         one dimension is bigger or equal than the search radius
 */
template<class T>
unsigned radiusToTreeLevel(T radius, T minRange)
{
    T radiusNormalized = radius / minRange;
    return unsigned(-log2(radiusNormalized));
}

/*! \brief findNeighbors of particle number \a id within radius
 *
 * Based on the Morton code of the input particle id, morton codes of neighboring
 * (implicit) octree nodes are computed in a first step, where the size of the
 * nodes is determined by the search radius. Each neighboring node corresponds to
 * a contigous range of particles in the coordinate arrays. The start and endpoints
 * of these ranges are determined by binary search and all particles within those ranges
 * are checked whether they lie within the search radius around the particle at id.
 *
 * \tparam T                   coordinate type, float or double
 * \tparam I                   Morton code type, uint32 uint64
 * \param[in] id               the index of the particle for which to look for neighbors
 * \param[in] x                particle x-coordinates
 * \param[in] y                particle y-coordinates
 * \param[in] z                particle z-coordinates
 * \param[in] radius           search radius
 * \param[in] boxMinRange      min_element(xmax-xmin, ymax-ymin, zmax-zmin)
 *                             of the global bounding box
 * \param[in] mortonCodes      Morton codes of all particles in x,y,z
 * \param[out] neighbors       output to store the neighbors
 * \param[out] neighborsCount  output to store the number of neighbors
 * \param[in] n                number of particles in x,y,z
 * \param[in] ngmax            maximum number of neighbors per particle
 */
template<class T, class I>
void findNeighbors(int id, const T* x, const T* y, const T* z, T radius, T boxMinRange,
                   const I* mortonCodes, int *neighbors, int *neighborsCount,
                   int n, int ngmax)
{
    T radiusSq = radius * radius;
    unsigned depth = radiusToTreeLevel(radius, boxMinRange);
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
