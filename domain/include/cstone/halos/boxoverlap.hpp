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
 * @brief (halo-)box overlap functionality
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/sfc/box.hpp"
#include "cstone/sfc/morton.hpp"

namespace cstone
{

//! @brief standard criterion for two ranges a-b and c-d to overlap, a<b and c<d
template<class T>
HOST_DEVICE_FUN constexpr bool overlapTwoRanges(T a, T b, T c, T d)
{
    assert(a<=b && c<=d);
    return b > c && d > a;
}

/*! @brief determine whether two ranges ab and cd overlap
 *
 * @tparam R  periodic range
 * @return    true or false
 *
 * Some restrictions apply, no input value can be further than R
 * from the periodic range.
 */
template<int R>
HOST_DEVICE_FUN constexpr bool overlapRange(int a, int b, int c, int d)
{
    assert(a >= -R);
    assert(a < R);
    assert(b > 0);
    assert(b <= 2*R);

    assert(c >= -R);
    assert(c < R);
    assert(d > 0);
    assert(d <= 2*R);

    return overlapTwoRanges(a,b,c,d) ||
           overlapTwoRanges(a+R, b+R, c, d) ||
           overlapTwoRanges(a, b, c+R, d+R);
}

/*! @brief check for overlap between a binary or octree node and a box in 3D space
 *
 * @tparam KeyType
 * @param prefix    Morton code node prefix, defines the corner of node
 *                  closest to origin. Also equals the lower Morton code bound
 *                  of the node.
 * @param length    Number of bits in the prefix to treat as the key. Defines
 *                  the Morton code range of the node.
 * @param box       3D coordinate range, defines an arbitrary box in space to
 *                  test for overlap.
 * @return          true or false
 *
 */
template <class KeyType>
HOST_DEVICE_FUN bool overlap(KeyType prefix, int length, const IBox& box)
{
    pair<int> xRange = idecodeMortonXRange(prefix, length);
    pair<int> yRange = idecodeMortonYRange(prefix, length);
    pair<int> zRange = idecodeMortonZRange(prefix, length);

    constexpr int maxCoord = 1u<<maxTreeLevel<KeyType>{};
    bool xOverlap = overlapRange<maxCoord>(xRange[0], xRange[1], box.xmin(), box.xmax());
    bool yOverlap = overlapRange<maxCoord>(yRange[0], yRange[1], box.ymin(), box.ymax());
    bool zOverlap = overlapRange<maxCoord>(zRange[0], zRange[1], box.zmin(), box.zmax());

    return xOverlap && yOverlap && zOverlap;
}

template <class KeyType>
HOST_DEVICE_FUN bool overlap(KeyType prefixBitKey, const IBox& box)
{
    int prefixLength = decodePrefixLength(prefixBitKey);
    return overlap(decodePlaceholderBit(prefixBitKey), prefixLength, box);
}

template <class KeyType>
HOST_DEVICE_FUN bool overlap(KeyType codeStart, KeyType codeEnd, const IBox& box)
{
    int level = treeLevel(codeEnd - codeStart);
    return overlap(codeStart, level*3, box);
}

/*! @brief Check whether a coordinate box is fully contained in a Morton code range
 *
 * @tparam KeyType   32- or 64-bit unsigned integer
 * @param codeStart  Morton code range start
 * @param codeEnd    Morton code range end
 * @param box        3D box with x,y,z integer coordinates in [0,2^maxTreeLevel<KeyType>{}-1]
 * @return           true if the box is fully contained within the specified Morton code range
 */
template <class KeyType>
HOST_DEVICE_FUN std::enable_if_t<std::is_unsigned_v<KeyType>, bool>
containedIn(KeyType codeStart, KeyType codeEnd, const IBox& box)
{
    // volume 0 boxes are not possible if makeHaloBox was used to generate it
    assert(box.xmin() < box.xmax());
    assert(box.ymin() < box.ymax());
    assert(box.zmin() < box.zmax());

    constexpr int pbcRange = 1u<<maxTreeLevel<KeyType>{};
    if (stl::min(stl::min(box.xmin(), box.ymin()), box.zmin()) < 0 ||
        stl::max(stl::max(box.xmax(), box.ymax()), box.zmax()) > pbcRange)
    {
        // any box that wraps around a PBC boundary cannot be contained within
        // any octree node, except the full root node
        return codeStart == 0 && codeEnd == nodeRange<KeyType>(0);
    }

    KeyType lowCode  = imorton3D<KeyType>(box.xmin(), box.ymin(), box.zmin());
    // we have to subtract 1 and use strict <, because we cannot generate
    // Morton codes for x,y,z >= 2^maxTreeLevel<KeyType>{} (2^10 or 2^21)
    KeyType highCode = imorton3D<KeyType>(box.xmax()-1, box.ymax()-1, box.zmax()-1);

    return (lowCode >= codeStart) && (highCode < codeEnd);
}

/*! @brief determine whether a binary/octree node (prefix, prefixLength) is fully contained in an SFC range
 *
 * @tparam KeyType       32- or 64-bit unsigned integer
 * @param prefix        lowest SFC code of the tree node
 * @param prefixLength  range of the tree node in bits,
 *                      corresponding SFC range is 2^(3*maxTreeLevel<KeyType>{} - prefixLength)
 * @param codeStart     start of the SFC range
 * @param codeEnd       end of the SFC range
 * @return
 */
template <class KeyType>
HOST_DEVICE_FUN inline std::enable_if_t<std::is_unsigned_v<KeyType>, bool>
containedIn(KeyType nodeStart, KeyType nodeEnd, KeyType codeStart, KeyType codeEnd)
{
    return !(nodeStart < codeStart || nodeEnd > codeEnd);
}

template <class KeyType>
HOST_DEVICE_FUN inline std::enable_if_t<std::is_unsigned_v<KeyType>, bool>
containedIn(KeyType prefixBitKey, KeyType codeStart, KeyType codeEnd)
{
    int prefixLength = decodePrefixLength(prefixBitKey);
    KeyType firstPrefix    = decodePlaceholderBit(prefixBitKey);
    KeyType secondPrefix   = firstPrefix + (KeyType(1) << (3*maxTreeLevel<KeyType>{} - prefixLength));
    return !(firstPrefix < codeStart || secondPrefix > codeEnd);
}

template <class KeyType>
HOST_DEVICE_FUN inline IBox makeIBox(KeyType mortonCodeStart, KeyType mortonCodeEnd)
{
    int prefixNBits = treeLevel(mortonCodeEnd -mortonCodeStart) * 3;

    pair<int> xrange = idecodeMortonXRange(mortonCodeStart, prefixNBits);
    pair<int> yrange = idecodeMortonYRange(mortonCodeStart, prefixNBits);
    pair<int> zrange = idecodeMortonZRange(mortonCodeStart, prefixNBits);

    return IBox(xrange[0], xrange[1], yrange[0], yrange[1], zrange[0], zrange[1]);
}

template<class KeyType>
HOST_DEVICE_FUN inline int addDelta(int value, int delta, bool pbc)
{
    constexpr int maxCoordinate = (1u << maxTreeLevel<KeyType>{});

    int temp = value + delta;
    if (pbc) return temp;
    else     return stl::min(stl::max(0, temp), maxCoordinate);
}

/*! @brief Construct a 3D box from an octree node plus halo range
 *
 * @tparam KeyType       32- or 64-bit unsigned integer
 * @param[in] codeStart  octree leaf node lower bound
 * @param[in] codeEnd    octree leaf node upper bound
 * @param[in] dx         extend X range by +- dx
 * @param[in] dy         extend Y range by +- dy
 * @param[in] dz         extend Z range by +- dz
 * @return               a box containing the integer coordinate ranges
 *                       of the input octree node extended by (dx,dy,dz)
 */
template <class KeyType>
HOST_DEVICE_FUN IBox makeHaloBox(KeyType codeStart, KeyType codeEnd, int dx, int dy, int dz,
                 bool pbcX = false, bool pbcY = false, bool pbcZ = false)
{
    IBox nodeBox = makeIBox(codeStart, codeEnd);

    return IBox(addDelta<KeyType>(nodeBox.xmin(), -dx, pbcX), addDelta<KeyType>(nodeBox.xmax(), dx, pbcX),
                addDelta<KeyType>(nodeBox.ymin(), -dy, pbcY), addDelta<KeyType>(nodeBox.ymax(), dy, pbcY),
                addDelta<KeyType>(nodeBox.zmin(), -dz, pbcZ), addDelta<KeyType>(nodeBox.zmax(), dz, pbcZ));
}

//! @brief create a box with specified radius around node delineated by codeStart/End
template <class CoordinateType, class RadiusType, class KeyType>
HOST_DEVICE_FUN IBox makeHaloBox(KeyType codeStart, KeyType codeEnd, RadiusType radius, const Box<CoordinateType>& box)
{
    // disallow boxes with no volume
    assert(codeEnd > codeStart);
    int dx = toNBitIntCeil<KeyType>(radius / (box.xmax() - box.xmin()));
    int dy = toNBitIntCeil<KeyType>(radius / (box.ymax() - box.ymin()));
    int dz = toNBitIntCeil<KeyType>(radius / (box.zmax() - box.zmin()));

    return makeHaloBox(codeStart, codeEnd, dx, dy, dz, box.pbcX(), box.pbcY(), box.pbcZ());
}

} // namespace cstone

