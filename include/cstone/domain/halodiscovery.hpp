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
#include "cstone/domain/domaindecomp.hpp"

namespace cstone
{

/*! @brief Compute halo node pairs
 *
 * @tparam CoordinateType      float or double
 * @tparam RadiusType          float or double, float is sufficient for 64-bit codes or less
 * @tparam I                   32- or 64-bit unsigned integer
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
template<class CoordinateType, class RadiusType, class I>
void findHalos(const std::vector<I>&             tree,
               const std::vector<RadiusType>&    interactionRadii,
               const Box<CoordinateType>&        box,
               TreeNodeIndex                     firstNode,
               TreeNodeIndex                     lastNode,
               std::vector<pair<TreeNodeIndex>>& haloPairs)
{
    std::vector<BinaryNode<I>> internalTree(nNodes(tree));
    createBinaryTree(tree.data(), nNodes(tree), internalTree.data());

    I lowestCode  = tree[firstNode];
    I highestCode = tree[lastNode];

    #pragma omp parallel
    {
        std::vector<pair<int>> threadHaloPairs;

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

                I collidingNodeStart = tree[collidingNodeIdx];
                I collidingNodeEnd   = tree[collidingNodeIdx + 1];

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

/*! @brief Compute send/receive node lists from halo pair node indices
 *
 * @param[in]  assignment       stores which rank owns which part of the SFC
 * @param[in]  haloPairs        list of mutually overlapping pairs of local/remote nodes
 * @param[out] incomingNodes    sorted list of halo nodes to be received,
 *                              grouped by source rank
 * @param[out] outgoingNodes    sorted list of internal nodes to be sent,
 *                              grouped by destination rank
 */
inline
void computeSendRecvNodeList(const SpaceCurveAssignment& assignment,
                             const std::vector<pair<TreeNodeIndex>>& haloPairs,
                             std::vector<std::vector<TreeNodeIndex>>& incomingNodes,
                             std::vector<std::vector<TreeNodeIndex>>& outgoingNodes)
{
    incomingNodes.resize(assignment.nRanks());
    outgoingNodes.resize(assignment.nRanks());

    for (auto& p : haloPairs)
    {
        // as defined in findHalos, the internal node index is stored first
        TreeNodeIndex internalNodeIdx = p[0];
        TreeNodeIndex remoteNodeIdx   = p[1];

        int remoteRank = assignment.findRank(remoteNodeIdx);

        incomingNodes[remoteRank].push_back(remoteNodeIdx);
        outgoingNodes[remoteRank].push_back(internalNodeIdx);
    }

    // remove duplicates in receiver list
    for (auto& v : incomingNodes)
    {
        std::sort(begin(v), end(v));
        auto unique_end = std::unique(begin(v), end(v));
        v.erase(unique_end, end(v));
    }

    // remove duplicates in sender list
    for (auto& v : outgoingNodes)
    {
        std::sort(begin(v), end(v));
        auto unique_end = std::unique(begin(v), end(v));
        v.erase(unique_end, end(v));
    }
}

} // namespace cstone