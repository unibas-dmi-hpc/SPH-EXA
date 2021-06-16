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
#include "cstone/util/gsl-lite.hpp"

namespace cstone
{

/*! @brief Compute halo node pairs
 *
 * @tparam KeyType             32- or 64-bit unsigned integer
 * @tparam RadiusType          float or double, float is sufficient for 64-bit codes or less
 * @tparam CoordinateType      float or double
 * @param tree                 cornerstone octree
 * @param interactionRadii     effective halo search radii per octree (leaf) node
 * @param box                  coordinate bounding box
 * @param firstNode            first node to consider as local
 * @param lastNode             last node to consider as local
 * @param[out] haloPairs       output list of halo node index pairs
 * @return
 *
 * A pair of indices (i,j) in [0...nNodes(tree)], is a halo pair for rank r if
 *   - tree[i] is in [firstNode:lastNode]
 *   - tree[j] is not in [firstNode:lastNode]
 *   - tree[i] enlarged by the search radius interactionRadii[i] overlaps with tree[j]
 *   - tree[j] enlarged by the search radius interactionRadii[j] overlaps with tree[i]
 *
 * This means that the first element in each index pair in @p haloPairs is the index of a
 * node (in @p tree) that must be sent out to another rank.
 * The second element of each pair is the index of a remote node not in [firstNode:lastNode].
 */
template<class KeyType, class RadiusType, class CoordinateType>
void findHalos(gsl::span<const KeyType>          tree,
               gsl::span<RadiusType>             interactionRadii,
               const Box<CoordinateType>&        box,
               TreeNodeIndex                     firstNode,
               TreeNodeIndex                     lastNode,
               std::vector<pair<TreeNodeIndex>>& haloPairs)
{
    std::vector<BinaryNode<KeyType>> internalTree(nNodes(tree));
    createBinaryTree(tree.data(), nNodes(tree), internalTree.data());

    KeyType lowestCode  = tree[firstNode];
    KeyType highestCode = tree[lastNode];

    #pragma omp parallel
    {
        std::vector<pair<TreeNodeIndex>> threadHaloPairs;

        // loop over all the nodes in range
        #pragma omp for
        for (TreeNodeIndex nodeIdx = firstNode; nodeIdx < lastNode; ++nodeIdx)
        {
            CollisionList collisions;
            RadiusType radius = interactionRadii[nodeIdx];

            IBox haloBox = makeHaloBox(tree[nodeIdx], tree[nodeIdx + 1], radius, box);

            // if the halo box is fully inside the assigned SFC range, we skip collision detection
            if (containedIn(lowestCode, highestCode, haloBox))
            {
                continue;
            }

            // find out with which other nodes in the octree that the node at nodeIdx
            // enlarged by the halo radius collides with
            findCollisions(internalTree.data(), tree.data(), collisions, haloBox, {lowestCode, highestCode});

            if (collisions.exhausted()) throw std::runtime_error("collision list exhausted\n");

            // collisions now has all nodes that collide with the haloBox around the node at tree[nodeIdx]
            // we only mark those nodes in collisions as halos if their haloBox collides with tree[nodeIdx]
            // i.e. we make sure that the local node (nodeIdx) is also a halo of the remote node
            for (std::size_t i = 0; i < collisions.size(); ++i)
            {
                TreeNodeIndex collidingNodeIdx = collisions[i];

                KeyType collidingNodeStart = tree[collidingNodeIdx];
                KeyType collidingNodeEnd   = tree[collidingNodeIdx + 1];

                IBox remoteNodeBox = makeHaloBox(collidingNodeStart, collidingNodeEnd,
                                                 interactionRadii[collidingNodeIdx], box);
                if (overlap(tree[nodeIdx], tree[nodeIdx + 1], remoteNodeBox))
                {
                    threadHaloPairs.emplace_back(nodeIdx, collidingNodeIdx);
                }
            }
        }
        #pragma omp critical
        {
            std::copy(begin(threadHaloPairs), end(threadHaloPairs), std::back_inserter(haloPairs));
        }
    }
}

/*! @brief mark halo nodes with flags
 *
 * @tparam KeyType               32- or 64-bit unsigned integer
 * @tparam RadiusType            float or double, float is sufficient for 64-bit codes or less
 * @tparam CoordinateType        float or double
 * @param[in]  tree              cornerstone octree leaves
 * @param[in]  binaryTree        matching binary tree on top of @p tree
 * @param[in]  interactionRadii  effective halo search radii per octree (leaf) node
 * @param[in]  box               coordinate bounding box
 * @param[in]  firstNode         first node to consider as local
 * @param[in]  lastNode          last node to consider as local
 * @param[out] collisionFlags    array of length nNodes(tree), each node that is a halo
 *                               from the perspective of [firstNode:lastNode] will be marked
 *                               with a non-zero value
 */
template<class KeyType, class RadiusType, class CoordinateType>
void findHalos(gsl::span<const KeyType> tree,
               gsl::span<const BinaryNode<KeyType>> binaryTree,
               gsl::span<RadiusType> interactionRadii,
               const Box<CoordinateType>& box,
               TreeNodeIndex firstNode,
               TreeNodeIndex lastNode,
               int* collisionFlags)
{
    KeyType lowestCode  = tree[firstNode];
    KeyType highestCode = tree[lastNode];

    // loop over all the nodes in range
    #pragma omp parallel for
    for (TreeNodeIndex nodeIdx = firstNode; nodeIdx < lastNode; ++nodeIdx)
    {
        RadiusType radius = interactionRadii[nodeIdx];
        IBox haloBox      = makeHaloBox(tree[nodeIdx], tree[nodeIdx + 1], radius, box);

        // if the halo box is fully inside the assigned SFC range, we skip collision detection
        if (containedIn(lowestCode, highestCode, haloBox)) { continue; }

        // mark all colliding node indices outside [lowestCode:highestCode]
        findCollisions(binaryTree.data(), tree.data(), collisionFlags, haloBox, {lowestCode, highestCode});
    }
}

} // namespace cstone