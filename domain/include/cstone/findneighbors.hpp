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

/*! @file
 * @brief  Find neighbors in Morton code sorted x,y,z arrays
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */


#pragma once

#include <cmath>

#include "cstone/primitives/stl.hpp"
#include "cstone/sfc/morton.hpp"

namespace cstone
{

/*! @brief compute squared distance, taking PBC into account
 *
 * Note that if box.pbc{X,Y,Z} is false, the result is identical to distancesq below.
 */
template<class T>
HOST_DEVICE_FUN constexpr T distanceSqPbc(T x1, T y1, T z1, T x2, T y2, T z2, const Box<T>& box)
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

//! @brief compute squared distance between to points in 3D
template<class T>
HOST_DEVICE_FUN constexpr T distancesq(T x1, T y1, T z1, T x2, T y2, T z2)
{
    T xx = x1 - x2;
    T yy = y1 - y2;
    T zz = z1 - z2;

    return xx * xx + yy * yy + zz * zz;
}

/*! @brief determines the octree subdivision layer at which the node edge length in
 *         one dimension is bigger or equal than the search radius
 */
template<class T>
HOST_DEVICE_FUN constexpr unsigned radiusToTreeLevel(T radius, T minRange)
{
    T radiusNormalized = stl::min(radius / minRange, T(1.0));
    return unsigned(-log2(radiusNormalized));
}

template<class KeyType>
HOST_DEVICE_FUN inline void storeCode(bool pbc, int* iNonPbc, int* iPbc, KeyType code, KeyType* boxes)
{
    if (pbc) boxes[--(*iPbc)]    = code;
    else     boxes[(*iNonPbc)++] = code;
}

/*! @brief find neighbor box codes
 *
 * @tparam T             float or double
 * @tparam KeyType             32- or 64-bit unsigned integer
 * @param[in]  xi        particle x coordinate
 * @param[in]  yi        particle y coordinate
 * @param[in]  zi        particle z coordinate
 * @param[in]  radiusSq  squared interaction radius of particle used to calculate d{x,y,z}{0,1}
 * @param[in]  bbox      global coordinate bounding box
 * @param[out] nCodes    output array for the neighbor box code, size has to be 27
 * @return               indexPair
 *
 * Box codes that were found without PBC are returned in nCodes[0:indexPair[0]],
 * while box codes found with PBC are returned in nCodes[indexPair[1]:27].
 *
 * The reason for distinguishing between non-PBC and PBC reachable boxes is that
 * non-PBC boxes can be searched for neighbors without PBC-adjusting the distances,
 * which is more than twice as expensive as plain direct distances.
 *
 * This function only adds a neighbor box if the sphere (xi,yi,zi)+-radius actually overlaps
 * with said box, which means that there are 26 different overlap checks.
 */
template<class T, class KeyType>
HOST_DEVICE_FUN pair<int> findNeighborBoxes(T xi, T yi, T zi, T radius, const Box<T>& bbox, KeyType* nCodes)
{
    constexpr int maxCoord = 1u<<maxTreeLevel<KeyType>{};
    // smallest octree cell edge length in unit cube
    constexpr T uL = T(1.) / maxCoord;

    T radiusSq = radius * radius;

    // level is the smallest tree subdivision level at which the node edge length is still bigger than radius
    unsigned level = radiusToTreeLevel(radius, bbox.minExtent());
    KeyType xyzCode = morton3D<KeyType>(xi, yi, zi, bbox);
    KeyType boxCode = enclosingBoxCode(xyzCode, level);

    auto [ixBox, iyBox, izBox] = decodeMorton(boxCode);
    T xBox = bbox.xmin() + ixBox * uL * bbox.lx();
    T yBox = bbox.ymin() + iyBox * uL * bbox.ly();
    T zBox = bbox.zmin() + izBox * uL * bbox.lz();

    int unitsPerBox = 1u<<(maxTreeLevel<KeyType>{} - level);
    T uLx = uL * bbox.lx() * unitsPerBox; // box length in x
    T uLy = uL * bbox.ly() * unitsPerBox; // box length in y
    T uLz = uL * bbox.lz() * unitsPerBox; // box length in z

    T dx0 = (xi - xBox) * (xi - xBox);
    T dx1 = (xi - xBox - uLx) * (xi - xBox - uLx);
    T dy0 = (yi - yBox) * (yi - yBox);
    T dy1 = (yi - yBox - uLy) * (yi - yBox - uLy);
    T dz0 = (zi - zBox) * (zi - zBox);
    T dz1 = (zi - (zBox + uLz)) * (zi - (zBox + uLz));

    bool hxd  = ixBox == 0;
    bool hxu  = ixBox >= maxCoord-unitsPerBox;
    bool hyd  = iyBox == 0;
    bool hyu  = iyBox >= maxCoord-unitsPerBox;
    bool hzd  = izBox == 0;
    bool hzu  = izBox >= maxCoord-unitsPerBox;
    bool stepXdown = !hxd || (bbox.pbcX() && level > 1);
    bool stepXup   = !hxu || (bbox.pbcX() && level > 1);
    bool stepYdown = !hyd || (bbox.pbcY() && level > 1);
    bool stepYup   = !hyu || (bbox.pbcY() && level > 1);
    bool stepZdown = !hzd || (bbox.pbcZ() && level > 1);
    bool stepZup   = !hzu || (bbox.pbcZ() && level > 1);

    int nBoxes  = 0;
    int iBoxPbc = 27;

    // If the radius is longer than the shortest bbox edge, but shorter than the longest
    // we have to look for neighbors with PBC if enabled.
    // This is a rare corner case that will never occur for sensible input data.
    bool periodicHomeBox = level == 0 && radius < bbox.maxExtent();

    // home box
    storeCode(periodicHomeBox, &nBoxes, &iBoxPbc, boxCode, nCodes);

    // X,Y,Z face touch
    if (dx0 < radiusSq && stepXdown)
        storeCode(hxd, &nBoxes, &iBoxPbc, mortonNeighbor(boxCode, level, -1, 0, 0), nCodes);
    if (dx1 < radiusSq && stepXup)
        storeCode(hxu, &nBoxes, &iBoxPbc, mortonNeighbor(boxCode, level,  1, 0, 0), nCodes);
    if (dy0 < radiusSq && stepYdown)
        storeCode(hyd, &nBoxes, &iBoxPbc, mortonNeighbor(boxCode, level, 0, -1, 0), nCodes);
    if (dy1 < radiusSq && stepYup)
        storeCode(hyu, &nBoxes, &iBoxPbc, mortonNeighbor(boxCode, level, 0,  1, 0), nCodes);
    if (dz0 < radiusSq && stepZdown)
        storeCode(hzd, &nBoxes, &iBoxPbc, mortonNeighbor(boxCode, level, 0, 0, -1), nCodes);
    if (dz1 < radiusSq && stepZup)
        storeCode(hzu, &nBoxes, &iBoxPbc, mortonNeighbor(boxCode, level, 0, 0, 1), nCodes);

    // XY edge touch
    if (dx0 + dy0 < radiusSq && stepXdown && stepYdown)
        storeCode(hxd || hyd, &nBoxes, &iBoxPbc, mortonNeighbor(boxCode, level, -1, -1, 0), nCodes);
    if (dx0 + dy1 < radiusSq && stepXdown && stepYup)
        storeCode(hxd || hyu, &nBoxes, &iBoxPbc, mortonNeighbor(boxCode, level, -1,  1, 0), nCodes);
    if (dx1 + dy0 < radiusSq && stepXup && stepYdown)
        storeCode(hxu || hyd, &nBoxes, &iBoxPbc, mortonNeighbor(boxCode, level,  1, -1, 0), nCodes);
    if (dx1 + dy1 < radiusSq && stepXup && stepYup)
        storeCode(hxu || hyu, &nBoxes, &iBoxPbc, mortonNeighbor(boxCode, level,  1,  1, 0), nCodes);

    // XZ edge touch
    if (dx0 + dz0 < radiusSq && stepXdown && stepZdown)
        storeCode(hxd || hzd, &nBoxes, &iBoxPbc, mortonNeighbor(boxCode, level, -1, 0, -1), nCodes);
    if (dx0 + dz1 < radiusSq && stepXdown && stepZup)
        storeCode(hxd || hzu, &nBoxes, &iBoxPbc, mortonNeighbor(boxCode, level, -1,  0, 1), nCodes);
    if (dx1 + dz0 < radiusSq && stepXup && stepZdown)
        storeCode(hxu || hzd, &nBoxes, &iBoxPbc, mortonNeighbor(boxCode, level,  1, 0, -1), nCodes);
    if (dx1 + dz1 < radiusSq && stepXup && stepZup)
        storeCode(hxu || hzu, &nBoxes, &iBoxPbc, mortonNeighbor(boxCode, level,  1,  0, 1), nCodes);

    // YZ edge touch
    if (dy0 + dz0 < radiusSq && stepYdown && stepZdown)
        storeCode(hyd || hzd, &nBoxes, &iBoxPbc, mortonNeighbor(boxCode, level, 0, -1, -1), nCodes);
    if (dy0 + dz1 < radiusSq && stepYdown && stepZup)
        storeCode(hyd || hzu, &nBoxes, &iBoxPbc, mortonNeighbor(boxCode, level,  0, -1, 1), nCodes);
    if (dy1 + dz0 < radiusSq && stepYup && stepZdown)
        storeCode(hyu || hzd, &nBoxes, &iBoxPbc, mortonNeighbor(boxCode, level,  0, 1, -1), nCodes);
    if (dy1 + dz1 < radiusSq && stepYup && stepZup)
        storeCode(hyu || hzu, &nBoxes, &iBoxPbc, mortonNeighbor(boxCode, level,  0,  1, 1), nCodes);

    // corner touches
    if (dx0 + dy0 + dz0 < radiusSq && stepXdown && stepYdown && stepZdown)
        storeCode(hxd || hyd || hzd, &nBoxes, &iBoxPbc, mortonNeighbor(boxCode, level, -1, -1, -1), nCodes);
    if (dx0 + dy0 + dz1 < radiusSq && stepXdown && stepYdown && stepZup)
        storeCode(hxd || hyd || hzu, &nBoxes, &iBoxPbc, mortonNeighbor(boxCode, level, -1, -1,  1), nCodes);
    if (dx0 + dy1 + dz0 < radiusSq && stepXdown && stepYup && stepZdown)
        storeCode(hxd || hyu || hzd, &nBoxes, &iBoxPbc, mortonNeighbor(boxCode, level, -1,  1, -1), nCodes);
    if (dx0 + dy1 + dz1 < radiusSq && stepXdown && stepYup && stepZup)
        storeCode(hxd || hyu || hzu, &nBoxes, &iBoxPbc, mortonNeighbor(boxCode, level, -1,  1,  1), nCodes);

    if (dx1 + dy0 + dz0 < radiusSq && stepXup && stepYdown && stepZdown)
        storeCode(hxu || hyd || hzd, &nBoxes, &iBoxPbc, mortonNeighbor(boxCode, level,  1, -1, -1), nCodes);
    if (dx1 + dy0 + dz1 < radiusSq && stepXup && stepYdown && stepZup)
        storeCode(hxu || hyd || hzu, &nBoxes, &iBoxPbc, mortonNeighbor(boxCode, level,  1, -1,  1), nCodes);
    if (dx1 + dy1 + dz0 < radiusSq && stepXup && stepYup && stepZdown)
        storeCode(hxu || hyu || hzd, &nBoxes, &iBoxPbc, mortonNeighbor(boxCode, level,  1,  1, -1), nCodes);
    if (dx1 + dy1 + dz1 < radiusSq && stepXup && stepYup && stepZup)
        storeCode(hxu || hyu || hzu, &nBoxes, &iBoxPbc, mortonNeighbor(boxCode, level,  1,  1,  1), nCodes);

    return pair<int>(nBoxes, iBoxPbc);
}

/*! @brief simple version
 *
 * @tparam T            float or double
 * @tparam KeyType            32- or 64-bit integer type
 * @param[in]  xi       x-coordinate of particle for which to do the neighbor search
 * @param[in]  yi       see xi
 * @param[in]  zi       see xi
 * @param[in]  radius   search radius
 * @param[in]  bbox     coordinate bounding box
 * @param[out] nCodes   output for boxes to search for neighbors of (xi, yi, zi), size 27
 * @return              a pair of indices, pair[0] is unused, pair[1] contains an index
 *                      into nCodes. The elements to search are nCodes[pair[1]:27]
 *
 * This simple version of findNeighborBoxes adds all (up to) 27 neighbor boxes, regardless
 * of whether or not they actually overlap with (xi,yi,zi)+-radius. All boxes are
 * return in the PBC-enabled part of @p nCodes, such that distanceSqPbc will be used to
 * calculate distances.
 */
template<class T, class KeyType>
HOST_DEVICE_FUN pair<int> findNeighborBoxesSimple(T xi, T yi, T zi, T radius, const Box<T>& bbox, KeyType* nCodes)
{
    // level is the smallest tree subdivision level at which the node edge length is still bigger than radius
    unsigned level = radiusToTreeLevel(radius, bbox.minExtent());
    KeyType xyzCode = morton3D<KeyType>(xi, yi, zi, bbox);
    KeyType boxCode = enclosingBoxCode(xyzCode, level);

    int ibox = 27;
    for (int dx = -1; dx < 2; ++dx)
        for (int dy = -1; dy < 2; ++dy)
            for (int dz = -1; dz < 2; ++dz)
            {
                KeyType searchBoxCode = mortonNeighbor(boxCode, level, dx, dy, dz);
                bool alreadyThere = false;
                for (int i = ibox; i < 27; ++i)
                {
                    if (nCodes[i] == searchBoxCode)
                        alreadyThere = true;
                }

                if (!alreadyThere)
                    nCodes[--ibox] = searchBoxCode;
            }

    return {0, ibox};
}

template<class KeyType, class T, class F>
HOST_DEVICE_FUN void searchBoxes(const KeyType* nCodes, int firstBox, int lastBox, const KeyType* mortonCodes, int n, int depth, int id,
                 const T* x, const T* y, const T* z, T radiusSq, int* neighbors, int* neighborsCount, int ngmax, F&& distance)
{
    T xi = x[id];
    T yi = y[id];
    T zi = z[id];

    int ngcount = *neighborsCount;
    for (int ibox = firstBox; ibox < lastBox; ++ibox)
    {
        KeyType neighbor = nCodes[ibox];
        int startIndex = stl::lower_bound(mortonCodes, mortonCodes + n, neighbor) - mortonCodes;
        int endIndex = stl::upper_bound(mortonCodes + startIndex, mortonCodes + n, neighbor + nodeRange<KeyType>(depth)) - mortonCodes;

        for (int j = startIndex; j < endIndex; ++j)
        {
            if (j == id) { continue; }

            if (ngcount == ngmax)
            {
                *neighborsCount = ngmax;
                return;
            }

            if (distance(xi, yi, zi, x[j], y[j], z[j]) < radiusSq) { neighbors[ngcount++] = j; }
        }
    }
    *neighborsCount = ngcount;
}


/*! @brief findNeighbors of particle number @p id within radius
 *
 * Based on the Morton code of the input particle id, morton codes of neighboring
 * (implicit) octree nodes are computed in a first step, where the size of the
 * nodes is determined by the search radius. Each neighboring node corresponds to
 * a contigous range of particles in the coordinate arrays. The start and endpoints
 * of these ranges are determined by binary search and all particles within those ranges
 * are checked whether they lie within the search radius around the particle at id.
 *
 * @tparam T                   coordinate type, float or double
 * @tparam KeyType             Morton code type, uint32 uint64
 * @param[in]  id              the index of the particle for which to look for neighbors
 * @param[in]  x               particle x-coordinates in Morton order
 * @param[in]  y               particle y-coordinates in Morton order
 * @param[in]  z               particle z-coordinates in Morton order
 * @param[in]  h               smoothing lengths (1/2 the search radius) in Morton order
 * @param[in]  box             coordinate bounding box that was used to calculate the Morton codes
 * @param[in]  mortonCodes     sorted Morton codes of all particles in x,y,z
 * @param[out] neighbors       output to store the neighbors
 * @param[out] neighborsCount  output to store the number of neighbors
 * @param[in]  n               number of particles in x,y,z
 * @param[in]  ngmax           maximum number of neighbors per particle
 */
template<class T, class KeyType>
HOST_DEVICE_FUN void findNeighbors(int id, const T* x, const T* y, const T* z, const T* h, const Box<T>& box,
                   const KeyType* mortonCodes, int *neighbors, int *neighborsCount,
                   int n, int ngmax)
{
    // SPH convention is search radius = 2 * h
    T radius       = 2 * h[id];
    T radiusSq     = radius * radius;

    // level is the smallest tree subdivision level at which the node edge length is still bigger than radius
    unsigned depth = radiusToTreeLevel(radius, box.minExtent());

    // load coordinates for particle #id
    T xi = x[id], yi = y[id], zi = z[id];

    KeyType neighborCodes[27];
    pair<int> boxCodeIndices = findNeighborBoxes(xi, yi, zi, radius, box, neighborCodes);
    //pair<int> boxCodeIndices = findNeighborBoxesSimple(xi, yi, zi, radius, box, neighborCodes);
    int       nBoxes         = boxCodeIndices[0];
    int       iBoxPbc        = boxCodeIndices[1];

    *neighborsCount = 0;

    // search non-PBC boxes
    searchBoxes(neighborCodes, 0, nBoxes, mortonCodes, n, depth, id, x, y, z, radiusSq, neighbors, neighborsCount,
                ngmax, [](T xi, T yi, T zi, T xj, T yj, T zj) { return distancesq(xi, yi, zi, xj, yj, zj); } );

    if (*neighborsCount == ngmax) { return; }

    // search PBC boxes
    searchBoxes(neighborCodes, iBoxPbc, 27, mortonCodes, n, depth, id, x, y, z, radiusSq, neighbors, neighborsCount, ngmax,
                [&box](T xi, T yi, T zi, T xj, T yj, T zj) { return distanceSqPbc(xi, yi, zi, xj, yj, zj, box); } );
}

} // namespace cstone
