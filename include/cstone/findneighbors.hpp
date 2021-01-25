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

#include "mortoncode.hpp"

namespace cstone
{

/*! \brief returns periodic shift dx, s.t. x + dx is folded into -l/2 to l/2
 *
 * @tparam T   float or double
 * @param x    input value
 * @param l    periodic length
 * @return     dx, such that x + dx is in [-l/2, l/2]
 */
template<class T>
static inline T pbcAdjust(T x, T l)
{
    return -l * std::rint(x/l);
}

//! \brief compute squared distance, taking PBC into account
template<class T>
static inline T distanceSqPbc(T x1, T y1, T z1, T x2, T y2, T z2, const Box<T>& box)
{
    T dx = x1 - x2;
    T dy = y1 - y2;
    T dz = z1 - z2;
    dx += box.pbcX() * pbcAdjust(dx, box.xmax() - box.xmin());
    dy += box.pbcY() * pbcAdjust(dy, box.ymax() - box.ymin());
    dz += box.pbcZ() * pbcAdjust(dz, box.zmax() - box.zmin());

    return dx * dx + dy * dy + dz * dz;
}

template<class T>
static inline T distancesq(T x1, T y1, T z1, T x2, T y2, T z2)
{
    T xx = x1 - x2;
    T yy = y1 - y2;
    T zz = z1 - z2;

    return xx * xx + yy * yy + zz * zz;
}

/*! \brief determines the octree subdivision layer at which the node edge length in
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
 * \param[in]  id              the index of the particle for which to look for neighbors
 * \param[in]  x               particle x-coordinates in Morton order
 * \param[in]  y               particle y-coordinates in Morton order
 * \param[in]  z               particle z-coordinates in Morton order
 * \param[in]  h               smoothing lengths (1/2 the search radius) in Morton order
 * \param[in]  box             coordinate bounding box that was used to calculate the Morton codes
 * \param[in]  mortonCodes     sorted Morton codes of all particles in x,y,z
 * \param[out] neighbors       output to store the neighbors
 * \param[out] neighborsCount  output to store the number of neighbors
 * \param[in]  n               number of particles in x,y,z
 * \param[in]  ngmax           maximum number of neighbors per particle
 */
template<class T, class I>
void findNeighbors(int id, const T* x, const T* y, const T* z, const T* h, const Box<T>& box,
                   const I* mortonCodes, int *neighbors, int *neighborsCount,
                   int n, int ngmax)
{
    // SPH convention is search radius = 2 * h
    T radius       = 2 * h[id];
    T radiusSq     = radius * radius;
    // depth is the smallest tree subdivision level at which the node edge length is still bigger than radius
    unsigned depth = radiusToTreeLevel(radius, box.minExtent());
    I mortonCode   = mortonCodes[id];

    std::array<I, 27> neighborCodes;

    // find neighboring boxes / octree nodes
    int nBoxes = 0;
    for (int dx = -1; dx < 2; ++dx)
        for (int dy = -1; dy < 2; ++dy)
            for (int dz = -1; dz < 2; ++dz)
                neighborCodes[nBoxes++] = mortonNeighbor(mortonCode, depth, dx, dy, dz,
                                                         box.pbcX(), box.pbcY(), box.pbcZ());

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

            if (distanceSqPbc(xi, yi, zi, x[j], y[j], z[j], box) < radiusSq)
            {
                neighbors[ngcount++] = j;
            }

            if (ngcount == ngmax) { break; }
        }
    }

    *neighborsCount = ngcount;
}

} // namespace cstone
