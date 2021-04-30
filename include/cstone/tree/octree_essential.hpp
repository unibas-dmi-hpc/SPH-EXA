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
 * @brief Generation of locally essential global octrees in cornerstone format
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

#include "cstone/cuda/annotation.hpp"
#include "cstone/sfc/morton.hpp"
#include "cstone/halos/boxoverlap.hpp"
#include "octree_internal.hpp"
#include "traversal.hpp"

namespace cstone
{

/*! @brief return the smallest distance squared between two points on the surface of the AABBs @p a and @p b
 *
 * @tparam T     float or double
 * @tparam I     32- or 64-bit unsigned integer
 * @param a      a box, specified with integer coordinates in [0:2^21]
 * @param b
 * @param box    floating point coordinate bounding box
 * @return       the square of the smallest distance between a and b
 */
template<class T, class I>
CUDA_HOST_DEVICE_FUN
T minDistanceSq(IBox a, IBox b, const Box<T>& box)
{
    constexpr size_t maxCoord = 1u<<maxTreeLevel<I>{};
    constexpr T unitLengthSq  = T(1.) / (maxCoord * maxCoord);

    size_t ux = stl::max(0, b.xmin() - a.xmax());
    size_t uy = stl::max(0, b.ymin() - a.ymax());
    size_t uz = stl::max(0, b.zmin() - a.zmax());

    size_t vx = stl::max(0, a.xmin() - b.xmax());
    size_t vy = stl::max(0, a.ymin() - b.ymax());
    size_t vz = stl::max(0, a.zmin() - b.zmax());

    // the maximum for any integer is 2^21-1, so we can safely square each of them
    return ((ux*ux + vx*vx)*box.lx()*box.lx() + (uy*uy + vy*vy)*box.ly()*box.ly() +
            (uz*uz + vz*vz)*box.lz()*box.lz()) * unitLengthSq;
}

//! @brief return longest edge length of box @p b
template<class T, class I>
CUDA_HOST_DEVICE_FUN
T nodeLength(IBox b, const Box<T>& box)
{
    constexpr int maxCoord = 1u<<maxTreeLevel<I>{};
    constexpr T unitLength = T(1.) / maxCoord;

    // IBoxes for octree nodes are assumed cubic, only box can be rectangular
    return (b.xmax() - b.xmin()) * unitLength * box.maxExtent();
}

/*! @brief evaluate minimum distance MAC
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
template<class T, class I>
CUDA_HOST_DEVICE_FUN
bool minDistanceMac(IBox a, IBox b, const Box<T>& box, float invThetaSq)
{
    T dsq = minDistanceSq<T, I>(a, b, box);
    // equivalent to "d > l / theta"
    T bLength = nodeLength<T, I>(b, box);
    return dsq > bLength * bLength * invThetaSq;
}

template<class T, class I>
CUDA_HOST_DEVICE_FUN
void markMacPerLeaf(TreeNodeIndex leafIdx, const Octree<I>& octree, const IBox* iboxes, const Box<T>& box,
                    float invThetaSq, char* markings)
{
    TreeNodeIndex octreeIdx = octree.toInternal(leafIdx);

    IBox target = makeIBox(octree.codeStart(octreeIdx), octree.codeEnd(octreeIdx));

    auto violatesMac = [target, iboxes, invThetaSq, &box](TreeNodeIndex idx)
    {
        return !minDistanceMac<T, I>(target, iboxes[idx], box, invThetaSq);
    };

    auto markIndex = [markings](TreeNodeIndex idx) { markings[idx] = 1; };

    traverse(octree, violatesMac, markIndex);
}

/*! @brief Mark all leaf nodes that fail the MAC paired with leaf nodes from a given range
 *
 * @tparam T                float or double
 * @tparam I                32- or 64-bit unsigned integer
 * @param[in]  octree       octree, including internal part
 * @param[in]  box          global coordinate bounding box
 * @param[in]  firstLeaf    first leaf index of the cornerstone tree used to build @p octree
 *                          to check for nodes failing the minimum distance Mac
 * @param[in]  lastLeaf     last leaf index
 * @param[in]  invThetaSq   1./theta^2
 * @param[out] markings     array of length @p octree.nLeafNodes, each position i outside [firstLeaf:lastLeaf]
 *                          will be set to 1, if the leaf node with index i fails the MAC paired with any
 *                          of the nodes in [firstLeaf:lastLeaf]
 */
template<class T, class I>
void markMac(const Octree<I>& octree, const Box<T>& box, TreeNodeIndex firstLeaf, TreeNodeIndex lastLeaf,
             float invThetaSq, char* markings)

{
    std::vector<IBox> treeBoxes(octree.nTreeNodes());

    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < octree.nTreeNodes(); ++i)
    {
        treeBoxes[i] = makeIBox(octree.codeStart(i), octree.codeEnd(i));
    }

    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = firstLeaf; i < lastLeaf; ++i)
    {
        markMacPerLeaf(i, octree, treeBoxes.data(), box, invThetaSq, markings);
    }
}

} // namespace cstone