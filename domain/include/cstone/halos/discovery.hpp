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
 * @brief  CPU driver for halo discovery using traversal of an internal binary radix tree
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <algorithm>
#include <vector>

#include "cstone/halos/btreetraversal.hpp"
#include "cstone/tree/traversal.hpp"
#include "cstone/util/gsl-lite.hpp"

namespace cstone
{

/*! @brief mark halo nodes with flags
 *
 * @tparam KeyType               32- or 64-bit unsigned integer
 * @tparam RadiusType            float or double, float is sufficient for 64-bit codes or less
 * @tparam CoordinateType        float or double
 * @param[in]  leaves            cornerstone octree leaves
 * @param[in]  binaryTree        matching binary tree on top of @p leaves
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
void findHalos(const KeyType* leaves,
               const BinaryNode<KeyType>* binaryTree,
               const RadiusType* interactionRadii,
               const Box<CoordinateType>& box,
               TreeNodeIndex firstNode,
               TreeNodeIndex lastNode,
               int* collisionFlags)
{
    KeyType lowestCode  = leaves[firstNode];
    KeyType highestCode = leaves[lastNode];

    // loop over all the nodes in range
    #pragma omp parallel for
    for (TreeNodeIndex nodeIdx = firstNode; nodeIdx < lastNode; ++nodeIdx)
    {
        RadiusType radius = interactionRadii[nodeIdx];
        IBox haloBox      = makeHaloBox(leaves[nodeIdx], leaves[nodeIdx + 1], radius, box);

        // if the halo box is fully inside the assigned SFC range, we skip collision detection
        if (containedIn(lowestCode, highestCode, haloBox)) { continue; }

        // mark all colliding node indices outside [lowestCode:highestCode]
        findCollisions(binaryTree, leaves, collisionFlags, haloBox, {lowestCode, highestCode});
    }
}

/*! @brief extract ranges of marked indices from a source array
 *
 * @tparam IntegralType  an integer type
 * @param source         array with quantities to extract, length N+1
 * @param flags          0 or 1 flags for index, length N
 * @param firstReqIdx    first index, permissible range: [0:N]
 * @param secondReqIdx   second index, permissible range: [0:N+1]
 * @return               vector (of pairs) of elements of @p source that span all
 *                       elements [firstReqIdx:secondReqIdx] of @p source that are
 *                       marked by @p flags
 *
 * Even indices mark the start of a range, uneven indices mark the end of the previous
 * range start. If two ranges are consecutive, they are fused into a single range.
 */
template<class IntegralType>
std::vector<IntegralType> extractMarkedElements(gsl::span<const IntegralType> source,
                                                gsl::span<const int> flags,
                                                TreeNodeIndex firstReqIdx,
                                                TreeNodeIndex secondReqIdx)
{
    std::vector<IntegralType> requestKeys;

    while (firstReqIdx != secondReqIdx)
    {
        // advance to first halo (or to secondReqIdx)
        while (firstReqIdx < secondReqIdx && flags[firstReqIdx] == 0) { firstReqIdx++; }

        // add one request key range
        if (firstReqIdx != secondReqIdx)
        {
            requestKeys.push_back(source[firstReqIdx]);
            // advance until not a halo or end of range
            while (firstReqIdx < secondReqIdx && flags[firstReqIdx] == 1) { firstReqIdx++; }
            requestKeys.push_back(source[firstReqIdx]);
        }
    }

    return requestKeys;
}

/*! @brief mark halo nodes with flags
 *
 * @tparam KeyType               32- or 64-bit unsigned integer
 * @tparam RadiusType            float or double, float is sufficient for 64-bit codes or less
 * @tparam CoordinateType        float or double
 * @param[in]  leaves            cornerstone octree leaves
 * @param[in]  binaryTree        matching binary tree on top of @p leaves
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
void findHalos(const Octree<KeyType>& octree,
               const RadiusType* interactionRadii,
               const Box<CoordinateType>& box,
               TreeNodeIndex firstNode,
               TreeNodeIndex lastNode,
               int* collisionFlags)
{
    auto leaves = octree.treeLeaves();
    KeyType lowestCode  = leaves[firstNode];
    KeyType highestCode = leaves[lastNode];

    auto markCollisions = [flags = collisionFlags](TreeNodeIndex i) { flags[i] = 1; };

    // loop over all the nodes in range
    #pragma omp parallel for
    for (TreeNodeIndex nodeIdx = firstNode; nodeIdx < lastNode; ++nodeIdx)
    {
        RadiusType radius = interactionRadii[nodeIdx];
        IBox haloBox      = makeHaloBox(leaves[nodeIdx], leaves[nodeIdx + 1], radius, box);

        // if the halo box is fully inside the assigned SFC range, we skip collision detection
        if (containedIn(lowestCode, highestCode, haloBox)) { continue; }

        auto overlaps = [lowestCode, highestCode, &octree, &haloBox](TreeNodeIndex idx)
        {
            KeyType octreeNode = octree.codeStart(idx);
            int level = octree.level(idx);
            return !containedIn(octreeNode, octreeNode + nodeRange<KeyType>(level), lowestCode, highestCode)
                   && overlap_(octreeNode, level, haloBox);
        };

        // mark all colliding node indices outside [lowestCode:highestCode]
        singleTraversal(octree, overlaps, markCollisions);
    }
}

} // namespace cstone
