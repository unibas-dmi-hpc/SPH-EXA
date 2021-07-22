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
 * A locally essential octree has a certain global resolution specified by a maximum
 * particle count per leaf node. In addition, it features a focus area defined as a
 * sub-range of the global space filling curve. In this focus sub-range, the resolution
 * can be higher, expressed through a smaller maximum particle count per leaf node.
 * Crucially, the resolution is also higher in the halo-areas of the focus sub-range.
 * These halo-areas can be defined as the overlap with the smoothing-length spheres around
 * the contained particles in the focus sub-range (SPH) or as the nodes whose opening angle
 * is too big to satisfy a multipole acceptance criterion from any perspective within the
 * focus sub-range (N-body).
 */

#pragma once

#include <vector>

#include "cstone/halos/boxoverlap.hpp"
#include "octree_internal.hpp"
#include "traversal.hpp"

namespace cstone
{

HOST_DEVICE_FUN
template<int Period>
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
    constexpr size_t maxCoord = 1u<<maxTreeLevel<KeyType>{};
    constexpr T unitLengthSq  = T(1.) / (maxCoord * maxCoord);

    size_t dx = rangeSeparation<maxCoord>(a.xmin(), a.xmax(), b.xmin(), b.xmax(), box.pbcX());
    size_t dy = rangeSeparation<maxCoord>(a.ymin(), a.ymax(), b.ymin(), b.ymax(), box.pbcY());
    size_t dz = rangeSeparation<maxCoord>(a.zmin(), a.zmax(), b.zmin(), b.zmax(), box.pbcZ());
    // the maximum for any integer is 2^21-1, so we can safely square each of them
    return ((dx*dx)*box.lx()*box.lx() + (dy*dy)*box.ly()*box.ly() +
            (dz*dz)*box.lz()*box.lz()) * unitLengthSq;
}

//! @brief return longest edge length of box @p b
template<class KeyType, class T>
HOST_DEVICE_FUN T nodeLength(IBox b, const Box<T>& box)
{
    constexpr int maxCoord = 1u<<maxTreeLevel<KeyType>{};
    constexpr T unitLength = T(1.) / maxCoord;

    // IBoxes for octree nodes are assumed cubic, only box can be rectangular
    return (b.xmax() - b.xmin()) * unitLength * box.maxExtent();
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
HOST_DEVICE_FUN bool minDistanceMac(IBox a, IBox b, const Box<T>& box, float invThetaSq)
{
    T dsq = minDistanceSq<KeyType>(a, b, box);
    // equivalent to "d > l / theta"
    T bLength = nodeLength<KeyType>(b, box);
    return dsq > bLength * bLength * invThetaSq;
}

//! @brief commutative version
template<class KeyType, class T>
HOST_DEVICE_FUN bool minDistanceMacMutual(IBox a, IBox b, const Box<T>& box, float invThetaSq)
{
    T dsq = minDistanceSq<KeyType>(a, b, box);
    // equivalent to "d > l / theta"
    T boxLength = stl::max(nodeLength<KeyType>(a, box), nodeLength<KeyType>(b, box));
    return dsq > boxLength * boxLength * invThetaSq;
}

template<class T, class KeyType>
HOST_DEVICE_FUN void markMacPerBox(IBox target, const Octree<KeyType>& octree, const Box<T>& box,
                   float invThetaSq, KeyType focusStart, KeyType focusEnd, char* markings)
{
    auto checkAndMarkMac = [target, &octree, &box, invThetaSq, focusStart, focusEnd, markings](TreeNodeIndex idx)
    {
        KeyType nodeStart = octree.codeStart(idx);
        KeyType nodeEnd   = octree.codeEnd(idx);
        // if the tree node with index idx is fully contained in the focus, we stop traversal
        if (containedIn(nodeStart, nodeEnd, focusStart, focusEnd)) { return false; }

        IBox sourceBox = makeIBox(nodeStart, nodeEnd);

        bool violatesMac = !minDistanceMac<KeyType>(target, sourceBox, box, invThetaSq);
        if (violatesMac) { markings[idx] = 1; }

        return violatesMac;
    };

    singleTraversal(octree, checkAndMarkMac, [](TreeNodeIndex) {});
}

/*! @brief Mark each node in an octree that fails the MAC paired with any node from a given focus SFC range
 *
 * @tparam T                float or double
 * @tparam KeyType          32- or 64-bit unsigned integer
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
        IBox target = makeIBox(focusCodes[i], focusCodes[i+1]);
        markMacPerBox(target, octree, box, invThetaSq, focusStart, focusEnd, markings);
    }
}

} // namespace cstone