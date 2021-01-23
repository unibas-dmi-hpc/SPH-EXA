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

template<int R>
int pbcAdjust(int x)
{
    return x - R * std::floor(double(x)/R);
}

template<int R>
bool overlapRange(int a, int b, int c, int d)
{
    int ap = pbcAdjust<R>(a);
    int bp = pbcAdjust<R>(b);
    int cp = pbcAdjust<R>(c);
    int dp = pbcAdjust<R>(d);

    bool fullRange = (std::abs(a-b) >= R) || (std::abs(c-d) >= R);

    bool c1 = bp > cp && dp > ap;
    bool c2 = ap > bp && cp > dp;
    bool c3 = bp > cp && (dp+R) > ap && (ap > bp);
    bool c4 = (bp+R) > cp && dp > ap && (cp > dp);

    return fullRange || c1 || c2 || c3 || c4;
}

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

/*! \brief Check whether a coordinate box is fully contained in a Morton code range
 *
 * @tparam I         32- or 64-bit unsigned integer
 * @param codeStart  Morton code range start
 * @param codeEnd    Morton code range end
 * @param box        3D box with x,y,z integer coordinates in [0,2^maxTreeLevel<I>{}-1]
 * @return           true if the box is fully contained within the specified Morton code range
 */
template <class I>
std::enable_if_t<std::is_unsigned_v<I>, bool>
containedIn(I codeStart, I codeEnd, const Box<int>& box)
{
    // volume 0 boxes are not possible if makeHaloBox was used to generate it
    assert(box.xmin() < box.xmax());
    assert(box.ymin() < box.ymax());
    assert(box.zmin() < box.zmax());

    I lowCode  = codeFromBox<I>(box.xmin(), box.ymin(), box.zmin(), maxTreeLevel<I>{});
    // we have to subtract 1 and use strict <, because we cannot generate
    // Morton codes for x,y,z >= 2^maxTreeLevel<I>{} (2^10 or 2^21)
    I highCode = codeFromBox<I>(box.xmax()-1, box.ymax()-1, box.zmax()-1, maxTreeLevel<I>{});

    return (lowCode >= codeStart) && (highCode < codeEnd);
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
Box<int> makeHaloBox(I codeStart, I codeEnd, int dx, int dy, int dz,
                     bool pbcX = false, bool pbcY = false, bool pbcZ = false)
{
    int prefixNBits = treeLevel(codeEnd - codeStart) * 3;

    pair<int> xrange = decodeXRange(codeStart, prefixNBits);
    pair<int> yrange = decodeYRange(codeStart, prefixNBits);
    pair<int> zrange = decodeZRange(codeStart, prefixNBits);

    constexpr int maxCoordinate = (1u << maxTreeLevel<I>{});

    // add halo range to the coordinate ranges of the node to be collided
    int xmin = (pbcX) ? xrange[0] - dx : std::max(0, xrange[0] - dx);
    int xmax = (pbcX) ? xrange[1] + dx : std::min(maxCoordinate, xrange[1] + dx);
    int ymin = (pbcY) ? yrange[0] - dy : std::max(0, yrange[0] - dy);
    int ymax = (pbcY) ? yrange[1] + dy : std::min(maxCoordinate, yrange[1] + dy);
    int zmin = (pbcZ) ? zrange[0] - dz : std::max(0, zrange[0] - dz);
    int zmax = (pbcZ) ? zrange[1] + dz : std::min(maxCoordinate, zrange[1] + dz);

    return Box<int>(xmin, xmax, ymin, ymax, zmin, zmax, pbcX, pbcY, pbcZ);
}

//! \brief create a box with specified radius around node delineated by codeStart/End
template <class I, class T>
Box<int> makeHaloBox(I codeStart, I codeEnd, T radius, const Box<T>& box)
{
    // disallow boxes with no volume
    assert(codeEnd > codeStart);
    int dx = detail::toNBitIntCeil<I>(radius / (box.xmax() - box.xmin()));
    int dy = detail::toNBitIntCeil<I>(radius / (box.ymax() - box.ymin()));
    int dz = detail::toNBitIntCeil<I>(radius / (box.zmax() - box.zmin()));

    return makeHaloBox(codeStart, codeEnd, dx, dy, dz, box.pbcX(), box.pbcY(), box.pbcZ());
}

} // namespace cstone