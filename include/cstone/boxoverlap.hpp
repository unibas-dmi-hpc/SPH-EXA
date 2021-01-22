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
 * \brief (halo-)box overlap functionality
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/box.hpp"
#include "cstone/mortoncode.hpp"


namespace cstone
{

/*! \brief check for overlap between a binary or octree node and a box in 3D space
 *
 * @tparam I
 * @param prefix            Morton code node prefix, defines the corner of node
 *                          closest to origin. Also equals the lower Morton code bound
 *                          of the node.
 * @param length            Number of bits in the prefix to treat as the key. Defines
 *                          the Morton code range of the node.
 * @param [x,y,z][min,max]  3D coordinate range, defines an arbitrary box in space to
 *                          test for overlap.
 * @return                  true or false
 *
 */
template <class I>
bool overlap(I prefix, int length, const Box<int>& box)
{
    pair<int> xRange = decodeXRange(prefix, length);
    pair<int> yRange = decodeYRange(prefix, length);
    pair<int> zRange = decodeZRange(prefix, length);

    bool xOverlap = box.xmax() > xRange[0] && xRange[1] > box.xmin();
    bool yOverlap = box.ymax() > yRange[0] && yRange[1] > box.ymin();
    bool zOverlap = box.zmax() > zRange[0] && zRange[1] > box.zmin();

    return xOverlap && yOverlap && zOverlap;
}

template <class I>
bool overlap(I codeStart, I codeEnd, const Box<int>& box)
{
    int level = treeLevel(codeEnd - codeStart);
    return overlap(codeStart, level*3, box);
}

/*! \brief Construct a 3D box from an octree node plus halo range
 *
 * @tparam I             32- or 64-bit unsigned integer
 * @param[in] codeStart  octree leaf node lower bound
 * @param[in] codeEnd    octree leaf node upper bound
 * @param[in] dx         extend X range by +- dx
 * @param[in] dy         extend Y range by +- dy
 * @param[in] dz         extend Z range by +- dz
 * @return               a box containing the integer coordinate ranges
 *                       of the input octree node extended by (dx,dy,dz)
 */
template <class I>
Box<int> makeHaloBox(I codeStart, I codeEnd, int dx, int dy, int dz)
{
    int prefixNBits = treeLevel(codeEnd - codeStart) * 3;

    pair<int> xrange = decodeXRange(codeStart, prefixNBits);
    pair<int> yrange = decodeYRange(codeStart, prefixNBits);
    pair<int> zrange = decodeZRange(codeStart, prefixNBits);

    constexpr int maxCoordinate = (1u << maxTreeLevel<I>{});

    // add halo range to the coordinate ranges of the node to be collided
    int xmin = std::max(0, xrange[0] - dx);
    int xmax = std::min(maxCoordinate, xrange[1] + dx);
    int ymin = std::max(0, yrange[0] - dy);
    int ymax = std::min(maxCoordinate, yrange[1] + dy);
    int zmin = std::max(0, zrange[0] - dz);
    int zmax = std::min(maxCoordinate, zrange[1] + dz);

    return Box<int>(xmin, xmax, ymin, ymax, zmin, zmax);
}

//! \brief create a box with specified radius around node delineated by codeStart/End
template <class I, class T>
Box<int> makeHaloBox(I codeStart, I codeEnd, T radius, const Box<T>& box)
{
    int dx = detail::toNBitIntCeil<I>(radius / (box.xmax() - box.xmin()));
    int dy = detail::toNBitIntCeil<I>(radius / (box.ymax() - box.ymin()));
    int dz = detail::toNBitIntCeil<I>(radius / (box.zmax() - box.zmin()));

    return makeHaloBox(codeStart, codeEnd, dx, dy, dz);
}

} // namespace cstone