#pragma once

#include "sfc/box.hpp"
#include "sfc/mortoncode.hpp"

//! \brief \file (halo-)box overlap functionality

namespace sphexa
{

/*! \brief check for overlap between a binary or octree node and an box 3D space
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
bool overlap(I prefix, int length, Box<int> box)
{
    pair<int> xRange = decodeXRange(prefix, length);
    pair<int> yRange = decodeYRange(prefix, length);
    pair<int> zRange = decodeZRange(prefix, length);

    bool xOverlap = box.xmax() > xRange[0] && xRange[1] > box.xmin();
    bool yOverlap = box.ymax() > yRange[0] && yRange[1] > box.ymin();
    bool zOverlap = box.zmax() > zRange[0] && zRange[1] > box.zmin();

    return xOverlap && yOverlap && zOverlap;
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
    int dx = detail::toNBitInt<I>(normalize(radius, box.xmin(), box.xmax()));
    int dy = detail::toNBitInt<I>(normalize(radius, box.ymin(), box.ymax()));
    int dz = detail::toNBitInt<I>(normalize(radius, box.zmin(), box.zmax()));

    return makeHaloBox(codeStart, codeEnd, dx, dy, dz);
}

} // namespace sphexa