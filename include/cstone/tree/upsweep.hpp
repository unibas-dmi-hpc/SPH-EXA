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
 * \brief  Generic octree upsweep procedure to calculate quantities for internal nodes from their children
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * Examples would be the calculation of particle counts for internal nodes for given leaf counts or
 * or the maximum smoothing length of any particle in the node.
 */

#pragma once

#include <atomic>

#include "octree_internal.hpp"

namespace cstone
{

/*! \brief atomically update a maximum value and return the previous maximum value
 *
 * @tparam T                     integer type
 * @param maximumValue[inout]    the maximum value to be atomically updated
 * @param newValue[in]           the value with which to compute the new maximum
 * @return                       the previous maximum value
 */
template<typename T>
T atomicMax(std::atomic<T>& maximumValue, const T& newValue) noexcept
{
    T previousValue = maximumValue;
    while(previousValue < newValue && !maximumValue.compare_exchange_weak(previousValue, newValue))
    {}
    return previousValue;
}

/*! \brief calculate distance to farthest leaf for each internal node in parallel
 *
 * @tparam I             32- or 64-bit integer type
 * @param octree[in]     an octree
 * @param depths[out]    array of length @a octree.nInternalNodes(), contains
 *                       the distance to the farthest leaf for each internal node.
 *                       The distance is equal to 1 for each node whose children are only leaves.
 */
template<class I>
void nodeDepth(const Octree<I>& octree, std::atomic<TreeNodeIndex>* depths)
{
    #pragma omp parallel for
    for (TreeNodeIndex i = 0; i < octree.nInternalNodes(); ++i)
    {
        int nLeafChildren = 0;
        for (int octant = 0; octant < 8; ++octant)
        {
            TreeNodeIndex child = octree.child(i, octant);
            if (octree.isLeaf(child))
            {
                nLeafChildren++;
            }
        }

        // all children are leaves - maximum depth is 1
        if (nLeafChildren == 8) { depths[i] = 1; }
        else                    { continue; }

        TreeNodeIndex nodeIndex = i;
        TreeNodeIndex depth     = 1;

        // race to the top
        do
        {   // ascend one level
            nodeIndex = octree.parent(nodeIndex);
            depth++;

            // set depths[nodeIndex] = max(depths[nodeIndex], depths) and store previous value
            // of depths[nodeIndex] in previousMax
            TreeNodeIndex previousMax = atomicMax(depths[nodeIndex], depth);
            if (previousMax >= depth)
            {
                // another thread already set a higher value for depths[nodeIndex], drop out of race
                break;
            }

        } while (nodeIndex != octree.parent(nodeIndex));
    }
}

template<class I>
void rewire(const OctreeNode<I>* oldNodes, const TreeNodeIndex* oldLeafParents,
            OctreeNode<I>* newNodes, TreeNodeIndex* newLeafParents, const TreeNodeIndex* rewireMap,
            TreeNodeIndex nInternalNodes)
{
    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex oldIndex = 0; oldIndex < nInternalNodes; ++oldIndex)
    {
        // node at <oldIndex> moves to <newIndex>
        TreeNodeIndex newIndex = rewireMap[oldIndex];

        OctreeNode<I> newNode = oldNodes[oldIndex];
        newNode.parent = rewireMap[newNode.parent];
        for (int octant = 0; octant < 8; ++octant)
        {
            TreeNodeIndex oldChild = newNode.child[octant];

            if (newNode.childType[octant] == OctreeNode<I>::leaf)
            {
                newLeafParents[oldChild] = newIndex;
            }
            else
            {
                newNode.child[octant] = rewireMap[oldChild];
            }
        }

        newNodes[newIndex] = newNode;
    }
}

template<class T>
using CombinationFunction = T (*)(const T*, const T*, const T*, const T*, const T*, const T*, const T*, const T*);


template<class T, class I>
void upsweep(const Octree<I>& octree, T* internalQuantities, const T* leafQuantities, CombinationFunction<T> combinationFunction)
{
    TreeNodeIndex nLeaves = octree.nLeaves();

    for (TreeNodeIndex i1 = 0; i1 < nLeaves/8; ++i1)
    {
        TreeNodeIndex nodeIdx = octree.parent(i1 * 8);
    }
}

} // namespace cstone