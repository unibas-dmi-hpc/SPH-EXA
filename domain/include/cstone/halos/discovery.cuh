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
 * @brief  GPU driver for halo discovery using traversal of an internal binary radix tree
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <algorithm>
#include <vector>

#include "cstone/halos/discovery.hpp"
#include "cstone/tree/btree.cuh"

namespace cstone
{

/*! @brief mark halo nodes with flags
 *
 * @tparam KeyType               32- or 64-bit unsigned integer
 * @tparam RadiusType            float or double, float is sufficient for 64-bit codes or less
 * @tparam CoordinateType        float or double
 * @param[in]  binaryTree        matching binary tree on top of @p leaves
 * @param[in]  leaves            cornerstone octree leaves
 * @param[in]  interactionRadii  effective halo search radii per octree (leaf) node
 * @param[in]  box               coordinate bounding box
 * @param[in]  firstNode         first leaf node index of @p leaves to consider as local
 * @param[in]  lastNode          last leaf node index of @p leaves to consider as local
 * @param[out] collisionFlags    array of length nNodes(leaves), each node that is a halo
 *                               from the perspective of [firstNode:lastNode] will be marked
 *                               with a non-zero value.
 *                               Note: does NOT reset non-colliding indices to 0, so @p collisionFlags
 *                               should be zero-initialized prior to calling this function.
 */
template<class KeyType, class RadiusType, class CoordinateType>
__global__ void findHalosKernel(const KeyType* leaves,
                                const BinaryNode<KeyType>* binaryTree,
                                const RadiusType* interactionRadii,
                                const Box<CoordinateType> box,
                                TreeNodeIndex firstNode,
                                TreeNodeIndex lastNode,
                                int* collisionFlags)
{
    unsigned tid     = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned nodeIdx = firstNode + tid;

    if (tid < lastNode - firstNode)
    {
        KeyType lowestCode  = leaves[firstNode];
        KeyType highestCode = leaves[lastNode];

        RadiusType radius = interactionRadii[nodeIdx];
        IBox haloBox      = makeHaloBox(leaves[nodeIdx], leaves[nodeIdx + 1], radius, box);

        // if the halo box is fully inside the assigned SFC range, we skip collision detection
        if (containedIn(lowestCode, highestCode, haloBox)) { return; }

        // mark all colliding node indices outside [lowestCode:highestCode]
        findCollisions(binaryTree, leaves, collisionFlags, haloBox, {lowestCode, highestCode});
    }
}

//! @brief convenience kernel wrapper
template<class KeyType, class RadiusType, class CoordinateType>
void findHalosGpu(const KeyType* leaves,
                  const BinaryNode<KeyType>* binaryTree,
                  const RadiusType* interactionRadii,
                  const Box<CoordinateType>& box,
                  TreeNodeIndex firstNode,
                  TreeNodeIndex lastNode,
                  int* collisionFlags)
{
    constexpr unsigned numThreads = 128;
    unsigned numBlocks            = iceil(lastNode - firstNode, numThreads);

    findHalosKernel<<<numBlocks, numThreads>>>(leaves, binaryTree, interactionRadii, box, firstNode, lastNode,
                                               collisionFlags);
}

} // namespace cstone
