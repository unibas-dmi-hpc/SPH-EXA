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
#include <cmath>

#include "bsearch.hpp"
#include "mortoncode.hpp"

namespace cstone
{

//! \brief compute squared distance, taking PBC into account
template<class T>
static inline T distanceSqPbc(T x1, T y1, T z1, T x2, T y2, T z2, const Box<T>& box)
{
    T dx = x1 - x2;
    T dy = y1 - y2;
    T dz = z1 - z2;
    // this folds d into the periodic range [-l/2, l/2] for each dimension if enabled
    dx -= box.pbcX() * box.lx() * std::rint(dx * box.ilx());
    dy -= box.pbcY() * box.ly() * std::rint(dy * box.ily());
    dz -= box.pbcZ() * box.lz() * std::rint(dz * box.ilz());

    return dx * dx + dy * dy + dz * dz;
}

//! \brief compute squared distance between to points in 3D
template<class T>
CUDA_HOST_DEVICE_FUN
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
CUDA_HOST_DEVICE_FUN
unsigned radiusToTreeLevel(T radius, T minRange)
{
    T radiusNormalized = stl::min(radius / minRange, T(1.0));
    return unsigned(-log2(radiusNormalized));
}

template<class T, class I>
int findNeighborBoxes(T dx0, T dx1, T dy0, T dy1, T dz0, T dz1,
                      I boxCode, int level, T radiusSq, bool pbcX, bool pbcY, bool pbcZ,
                      I* nCodes)
{
    int nBoxes = 0;

    // home box
    nCodes[nBoxes++] = boxCode;

    // X,Y,Z face touch
    if (dx0 < radiusSq)
        nCodes[nBoxes++] = mortonNeighbor(boxCode, level, -1, 0, 0, pbcX, pbcY, pbcZ);
    if (dx1 < radiusSq)
        nCodes[nBoxes++] = mortonNeighbor(boxCode, level,  1, 0, 0, pbcX, pbcY, pbcZ);
    if (dy0 < radiusSq)
        nCodes[nBoxes++] = mortonNeighbor(boxCode, level, 0, -1, 0, pbcX, pbcY, pbcZ);
    if (dy1 < radiusSq)
        nCodes[nBoxes++] = mortonNeighbor(boxCode, level, 0,  1, 0, pbcX, pbcY, pbcZ);
    if (dz0 < radiusSq)
        nCodes[nBoxes++] = mortonNeighbor(boxCode, level, 0, 0, -1, pbcX, pbcY, pbcZ);
    if (dz1 < radiusSq)
        nCodes[nBoxes++] = mortonNeighbor(boxCode, level, 0, 0, 1, pbcX, pbcY, pbcZ);

    // XY edge touch
    if (dx0 + dy0 < radiusSq)
        nCodes[nBoxes++] = mortonNeighbor(boxCode, level, -1, -1, 0, pbcX, pbcY, pbcZ);
    if (dx0 + dy1 < radiusSq)
        nCodes[nBoxes++] = mortonNeighbor(boxCode, level, -1,  1, 0, pbcX, pbcY, pbcZ);
    if (dx1 + dy0 < radiusSq)
        nCodes[nBoxes++] = mortonNeighbor(boxCode, level,  1, -1, 0, pbcX, pbcY, pbcZ);
    if (dx1 + dy1 < radiusSq)
        nCodes[nBoxes++] = mortonNeighbor(boxCode, level,  1,  1, 0, pbcX, pbcY, pbcZ);

    // XZ edge touch
    if (dx0 + dz0 < radiusSq)
        nCodes[nBoxes++] = mortonNeighbor(boxCode, level, -1, 0, -1, pbcX, pbcY, pbcZ);
    if (dx0 + dz1 < radiusSq)
        nCodes[nBoxes++] = mortonNeighbor(boxCode, level, -1,  0, 1, pbcX, pbcY, pbcZ);
    if (dx1 + dz0 < radiusSq)
        nCodes[nBoxes++] = mortonNeighbor(boxCode, level,  1, 0, -1, pbcX, pbcY, pbcZ);
    if (dx1 + dz1 < radiusSq)
        nCodes[nBoxes++] = mortonNeighbor(boxCode, level,  1,  0, 1, pbcX, pbcY, pbcZ);

    // YZ edge touch
    if (dy0 + dz0 < radiusSq)
        nCodes[nBoxes++] = mortonNeighbor(boxCode, level, 0, -1, -1, pbcX, pbcY, pbcZ);
    if (dy0 + dz1 < radiusSq)
        nCodes[nBoxes++] = mortonNeighbor(boxCode, level, 0, -1, 1, pbcX, pbcY, pbcZ);
    if (dy1 + dz0 < radiusSq)
        nCodes[nBoxes++] = mortonNeighbor(boxCode, level,  0, 1, -1, pbcX, pbcY, pbcZ);
    if (dy1 + dz1 < radiusSq)
        nCodes[nBoxes++] = mortonNeighbor(boxCode, level,  0,  1, 1, pbcX, pbcY, pbcZ);

    // corner touches
    if (dx0 + dy0 + dz0 < radiusSq)
        nCodes[nBoxes++] = mortonNeighbor(boxCode, level, -1, -1, -1, pbcX, pbcY, pbcZ);
    if (dx0 + dy0 + dz1 < radiusSq)
        nCodes[nBoxes++] = mortonNeighbor(boxCode, level, -1, -1, 1, pbcX, pbcY, pbcZ);
    if (dx0 + dy1 + dz0 < radiusSq)
        nCodes[nBoxes++] = mortonNeighbor(boxCode, level, -1,  1, -1, pbcX, pbcY, pbcZ);
    if (dx0 + dy1 + dz1 < radiusSq)
        nCodes[nBoxes++] = mortonNeighbor(boxCode, level, -1,  1,  1, pbcX, pbcY, pbcZ);

    if (dx1 + dy0 + dz0 < radiusSq)
        nCodes[nBoxes++] = mortonNeighbor(boxCode, level,  1, -1, -1, pbcX, pbcY, pbcZ);
    if (dx1 + dy0 + dz1 < radiusSq)
        nCodes[nBoxes++] = mortonNeighbor(boxCode, level,  1, -1, 1, pbcX, pbcY, pbcZ);
    if (dx1 + dy1 + dz0 < radiusSq)
        nCodes[nBoxes++] = mortonNeighbor(boxCode, level,  1,  1, -1, pbcX, pbcY, pbcZ);
    if (dx1 + dy1 + dz1 < radiusSq)
        nCodes[nBoxes++] = mortonNeighbor(boxCode, level,  1,  1,  1, pbcX, pbcY, pbcZ);

    return nBoxes;

    // unoptimized version
    //int nBoxes = 0;
    //for (int dx = -1; dx < 2; ++dx)
    //    for (int dy = -1; dy < 2; ++dy)
    //        for (int dz = -1; dz < 2; ++dz)
    //        {
    //            nCodes[nBoxes++] = mortonNeighbor(boxCode, level, dx, dy, dz,
    //                                              box.pbcX(), box.pbcY(), box.pbcZ());
    //        }
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
CUDA_HOST_DEVICE_FUN
void findNeighbors(int id, const T* x, const T* y, const T* z, const T* h, const Box<T>& box,
                   const I* mortonCodes, int *neighbors, int *neighborsCount,
                   int n, int ngmax)
{
    // SPH convention is search radius = 2 * h
    T radius       = 2 * h[id];
    T radiusSq     = radius * radius;

    // level is the smallest tree subdivision level at which the node edge length is still bigger than radius
    unsigned depth = radiusToTreeLevel(radius, box.minExtent());

    I boxCode = enclosingBoxCode(mortonCodes[id], depth);
    T xBox    = box.xmin() + decodeMortonX(boxCode) * uL * box.lx();
    T yBox    = box.ymin() + decodeMortonY(boxCode) * uL * box.ly();
    T zBox    = box.zmin() + decodeMortonZ(boxCode) * uL * box.lz();

    *neighborsCount = 0;

    // search non-PBC boxes
    searchBoxes(nCodes, 0, nBoxes, mortonCodes, n, depth, id, x, y, z, radiusSq, neighbors, neighborsCount, ngmax,
                [](T xi, T yi, T zi, T xj, T yj, T zj) { return distancesq(xi, yi, zi, xj, yj, zj); } );

    if (*neighborsCount == ngmax) { return; }

    T dxiBx0 = (xi - xBox) * (xi - xBox);
    T dxiBx1 = (xi - xBox - uLx) * (xi - xBox - uLx);
    T dyiBy0 = (yi - yBox) * (yi - yBox);
    T dyiBy1 = (yi - yBox - uLy) * (yi - yBox - uLy);
    T dziBz0 = (zi - zBox) * (zi - zBox);
    T dziBz1 = (zi - (zBox + uLz)) * (zi - (zBox + uLz));

    I nCodes[27]; // neighborCodes
    int nBoxes = findNeighborBoxes(dxiBx0, dxiBx1, dyiBy0, dyiBy1, dziBz0, dziBz1, boxCode, depth, radiusSq,
                                   box.pbcX(), box.pbcY(), box.pbcZ(), nCodes);

    std::sort(nCodes, nCodes + nBoxes);
    nBoxes = std::unique(nCodes, nCodes + nBoxes) - nCodes;

    int ngcount = 0;
    for (int ibox = 0; ibox < nBoxes; ++ibox)
    {
        I neighbor     = nCodes[ibox];
        int startIndex = std::lower_bound(mortonCodes, mortonCodes + n, neighbor) - mortonCodes;
        int endIndex   = std::upper_bound(mortonCodes + startIndex, mortonCodes + n,
                                          neighbor + nodeRange<I>(depth)) - mortonCodes;

    //    for (int j = startIndex; j < endIndex; ++j)
    //    {
    //        if (j == id) { continue; }

            if (distanceSqPbc(xi, yi, zi, x[j], y[j], z[j], box) < radiusSq)
            //if (distancesq(xi, yi, zi, x[j], y[j], z[j]) < radiusSq)
            {
                neighbors[ngcount++] = j;
            }

    //        if (ngcount == ngmax) { *neighborsCount = ngmax; return; }
    //    }
    //}

    //*neighborsCount = ngcount;
}

} // namespace cstone
