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
 * @brief Evaluate multipole acceptance criteria (MAC) on octree nodes
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 */

#pragma once

#include <vector>

#include "boxoverlap.hpp"
#include "cstone/tree/octree_internal.hpp"
#include "cstone/traversal/traversal.hpp"

namespace cstone
{

template<int Period>
HOST_DEVICE_FUN
constexpr int rangeSeparation(int a, int b, int c, int d, bool pbc)
{
    assert(a < b && c < d);
    int cb = c - b;
    int ad = a - d;
    int cbPbc = (pbc) ? pbcDistance<Period>(cb) : cb;
    int adPbc = (pbc) ? pbcDistance<Period>(ad) : ad;
    return (cb >= 0 || ad >= 0) * stl::min(stl::abs(cbPbc), stl::abs(adPbc));
}

/*! @brief return the smallest distance squared between two points on the surface of the AABBs @p a and @p b
 *
 * @tparam T        float or double
 * @tparam KeyType  32- or 64-bit unsigned integer
 * @param a         a box, specified with integer coordinates in [0:2^21]
 * @param b
 * @param box       floating point coordinate bounding box
 * @return          the square of the smallest distance between a and b
 */
template<class KeyType, class T>
HOST_DEVICE_FUN T minDistanceSq(IBox a, IBox b, const Box<T>& box)
{
    constexpr size_t mc = maxCoord<KeyType>{};
    constexpr T unitLengthSq  = T(1.) / (mc * mc);

    size_t dx = rangeSeparation<mc>(a.xmin(), a.xmax(), b.xmin(), b.xmax(), box.pbcX());
    size_t dy = rangeSeparation<mc>(a.ymin(), a.ymax(), b.ymin(), b.ymax(), box.pbcY());
    size_t dz = rangeSeparation<mc>(a.zmin(), a.zmax(), b.zmin(), b.zmax(), box.pbcZ());
    // the maximum for any integer is 2^21-1, so we can safely square each of them
    return ((dx*dx)*box.lx()*box.lx() + (dy*dy)*box.ly()*box.ly() +
            (dz*dz)*box.lz()*box.lz()) * unitLengthSq;
}

//! @brief return longest edge length of box @p b
template<class KeyType, class T>
HOST_DEVICE_FUN T nodeLength(IBox b, const Box<T>& box)
{
    constexpr T unitLength = T(1.) / maxCoord<KeyType>{};

    // IBoxes for octree nodes are assumed cubic, only box can be rectangular
    return (b.xmax() - b.xmin()) * unitLength * box.maxExtent();
}

//! @brief returns the smallest PBC distance of point (x,y,z) to box b
template<class KeyType, class T>
HOST_DEVICE_FUN T minDistance(T x, T y, T z, IBox b, const Box<T>& box)
{
    constexpr T uL = T(1.) / maxCoord<KeyType>{};

    T dxa = x - b.xmin() * uL * box.lx();
    T dxb = x - b.xmax() * uL * box.lx();
    T dya = y - b.ymin() * uL * box.ly();
    T dyb = y - b.ymax() * uL * box.ly();
    T dza = z - b.zmin() * uL * box.lz();
    T dzb = z - b.zmax() * uL * box.lz();

    // this folds d into the periodic range [-l/2, l/2] if enabled
    dxa -= box.pbcX() * box.lx() * std::rint(dxa * box.ilx());
    dxb -= box.pbcX() * box.lx() * std::rint(dxb * box.ilx());
    dya -= box.pbcY() * box.ly() * std::rint(dya * box.ily());
    dyb -= box.pbcY() * box.ly() * std::rint(dyb * box.ily());
    dza -= box.pbcZ() * box.lz() * std::rint(dza * box.ilz());
    dzb -= box.pbcZ() * box.lz() * std::rint(dzb * box.ilz());

    return sqrt(stl::min(dxa*dxa, dxb*dxb) + stl::min(dya*dya, dyb*dyb) + stl::min(dza*dza, dzb*dzb));
}

/*! @brief vector multipole acceptance criterion
 *
 * @tparam KeyType   unsigned 32- or 64-bit integer type
 * @tparam T         float or double
 * @param  comx      center of gravity x-coordinate
 * @param  comy      center of gravity y-coordinate
 * @param  comz      center of gravity z-coordinate
 * @param  source    source integer coordinate box
 * @param  target    target integer coordinate box
 * @param  box       global coordinate bounding box, contains PBC information
 * @param  theta     accuracy parameter
 * @return           true if criterion fulfilled (cell does not need to be opened)
 *
 * Evaluates  d > l/theta + delta with:
 *  d     -> minimal distance of target box to source center of mass
 *  l     -> edge length of source box
 *  delta -> distance from geometric source center to source center mass
 */
template<class KeyType, class T>
HOST_DEVICE_FUN bool vectorMac(T comx, T comy, T comz, const IBox& source, const IBox& target, const Box<T>& box,
                               float theta)
{
    constexpr T uL = T(1.) / maxCoord<KeyType>{};

    // minimal distance from source-center-mass to target box
    T distanceToCom = minDistance<KeyType>(comx, comy, comz, target, box);
    T sourceLength  = nodeLength<KeyType>(source, box);

    // geometric center of source
    int halfCube = (source.xmax() - source.xmin()) / 2;
    T xsc = (source.xmin() + halfCube) * uL * box.lx();
    T ysc = (source.ymin() + halfCube) * uL * box.ly();
    T zsc = (source.zmin() + halfCube) * uL * box.lz();

    T dxsc = xsc - comx;
    T dysc = ysc - comy;
    T dzsc = zsc - comz;

    T distanceCom2SourceCenter = sqrt(dxsc * dxsc + dysc * dysc + dzsc * dzsc);

    return distanceToCom > sourceLength/theta + distanceCom2SourceCenter;
}

/*! @brief evaluate minimum distance MAC, non-commutative version
 *
 * @param a            target cell
 * @param b            source cell
 * @param box          coordinate bounding box
 * @param invThetaSq   inverse theta squared
 * @return             true if MAC fulfilled, false otherwise
 *
 * Note: Mac is valid for any point in a w.r.t to box b, therefore only the
 * size of b is relevant.
 */
template<class KeyType, class T>
HOST_DEVICE_FUN bool minDistanceMac(const IBox& a, const IBox& b, const Box<T>& box, float invThetaSq)
{
    T dsq = minDistanceSq<KeyType>(a, b, box);
    // equivalent to "d > l / theta"
    T bLength = nodeLength<KeyType>(b, box);
    return dsq > bLength * bLength * invThetaSq;
}

//! @brief commutative version
template<class KeyType, class T>
HOST_DEVICE_FUN bool minDistanceMacMutual(const IBox& a, const IBox& b, const Box<T>& box, float invThetaSq)
{
    T dsq = minDistanceSq<KeyType>(a, b, box);
    // equivalent to "d > l / theta"
    T boxLength = stl::max(nodeLength<KeyType>(a, box), nodeLength<KeyType>(b, box));
    return dsq > boxLength * boxLength * invThetaSq;
}

//! @brief mark all nodes of @p octree (leaves and internal) that fail the MAC w.r.t to @p target
template<class T, class KeyType>
void markMacPerBox(IBox target, const Octree<KeyType>& octree, const Box<T>& box,
                   float invThetaSq, KeyType focusStart, KeyType focusEnd, char* markings)
{
    auto checkAndMarkMac = [target, &octree, &box, invThetaSq, focusStart, focusEnd, markings](TreeNodeIndex idx)
    {
        KeyType nodeStart = octree.codeStart(idx);
        KeyType nodeEnd   = octree.codeEnd(idx);
        // if the tree node with index idx is fully contained in the focus, we stop traversal
        if (containedIn(nodeStart, nodeEnd, focusStart, focusEnd)) { return false; }

        IBox sourceBox = sfcIBox(sfcKey(nodeStart), octree.level(idx));

        bool violatesMac = !minDistanceMac<KeyType>(target, sourceBox, box, invThetaSq);
        if (violatesMac) { markings[idx] = 1; }

        return violatesMac;
    };

    singleTraversal(octree, checkAndMarkMac, [](TreeNodeIndex) {});
}

/*! @brief Mark each node in an octree that fails the MAC paired with any node from a given focus SFC range
 *
 * @tparam     T            float or double
 * @tparam     KeyType      32- or 64-bit unsigned integer
 * @param[in]  octree       octree, including internal part
 * @param[in]  box          global coordinate bounding box
 * @param[in]  focusStart   lower SFC focus code
 * @param[in]  focusEnd     upper SFC focus code
 * @param[in]  invThetaSq   1./theta^2
 * @param[out] markings     array of length @p octree.numTreeNodes(), each position i
 *                          will be set to 1, if the node of @p octree with index i fails the MAC paired with
 *                          any node contained in the focus range [focusStart:focusEnd]
 */
template<class T, class KeyType>
void markMac(const Octree<KeyType>& octree, const Box<T>& box, KeyType focusStart, KeyType focusEnd,
             float invThetaSq, char* markings)

{
    std::fill(markings, markings + octree.numTreeNodes(), 0);

    // find the minimum possible number of octree node boxes to cover the entire focus
    TreeNodeIndex numFocusBoxes = spanSfcRange(focusStart, focusEnd);
    std::vector<KeyType> focusCodes(numFocusBoxes + 1);
    spanSfcRange(focusStart, focusEnd, focusCodes.data());
    focusCodes.back() = focusEnd;

    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < numFocusBoxes; ++i)
    {
        IBox target = sfcIBox(sfcKey(focusCodes[i]), sfcKey(focusCodes[i + 1]));
        markMacPerBox(target, octree, box, invThetaSq, focusStart, focusEnd, markings);
    }
}

} // namespace cstone
