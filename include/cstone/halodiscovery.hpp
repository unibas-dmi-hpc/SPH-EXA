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
 * \brief  CPU driver for halo discovery using traversal of an internal binary radix tree
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <algorithm>
#include <vector>

#include "cstone/btreetraversal.hpp"
#include "cstone/domaindecomp.hpp"

namespace cstone
{

/*! \brief Compute halo node pairs
 *
 * @tparam I                   32- or 64-bit unsigned integer
 * @tparam T                   float or double
 * @param tree                 cornerstone octree
 * @param interactionRadii     effective halo search radii per octree (leaf) node
 * @param box                  coordinate bounding box
 * @param assignment           list if Morton code ranges assignments per rank
 * @param rank                 compute pairs from perspective of \a rank
 * @param[out] haloPairs       output list of halo node index pairs
 * @return
 *
 * A pair of indices (i,j) in [0...nNodes(tree)], is a halo pair for rank r if
 *   - tree[i] is assigned to rank r
 *   - tree[j] is not assigned to rank r
 *   - tree[i] enlarged by the search radius interactionRadii[i] overlaps with tree[j]
 *   - tree[j] enlarged by the search radius interactionRadii[j] overlaps with tree[i]
 *
 * This means that the first element in each index pair in \a haloPairs is the index of a
 * node (in \a tree) that belongs to rank \a rank and must be sent out to another rank.
 *
 * The second element of each pair is the index of a remote node that is a halo for rank \a rank.
 * We can easily find the source rank of the halo with binary search in the space curve assignment.
 * The source rank of the halo is also the destination where the internal node referenced in the first
 * pair element must be sent to.
 */
template<class I, class T>
void findHalos(const std::vector<I>&           tree,
               const std::vector<T>&           interactionRadii,
               const Box<T>&                   box,
               const SpaceCurveAssignment<I>&  assignment,
               int                             rank,
               std::vector<pair<int>>&         haloPairs)
{
    std::vector<BinaryNode<I>> internalTree = createInternalTree(tree);

    // go through all ranges assigned to rank
    for (int range = 0; range < assignment.nRanges(rank); ++range)
    {
        int firstNode = std::lower_bound(begin(tree), end(tree), assignment.rangeStart(rank, range)) - begin(tree);
        int lastNode  = std::lower_bound(begin(tree), end(tree), assignment.rangeEnd(rank, range)) - begin(tree);

        // loop over all the nodes in range
        for (int nodeIdx = firstNode; nodeIdx < lastNode; ++nodeIdx)
        {
            CollisionList collisions;
            T radius = interactionRadii[nodeIdx];

            // find out with which other nodes in the octree that the node at nodeIdx
            // enlarged by the halo radius collides with
            Box<int> haloBox = makeHaloBox(tree[nodeIdx], tree[nodeIdx + 1], radius, box);

            // TODO skip collision detection if is the halo box is fully contained in the assigned range
            findCollisions(internalTree.data(), tree.data(), collisions, haloBox);

            if (collisions.exhausted()) throw std::runtime_error("collision list exhausted\n");

            // Go through all colliding nodes to determine which of them fall into a part of the SFC
            // that is not assigned to the executing rank. These nodes will be marked as halos.
            for (int i = 0; i < collisions.size(); ++i)
            {
                int collidingNodeIdx = collisions[i];

                I collidingNodeStart = tree[collidingNodeIdx];
                I collidingNodeEnd   = tree[collidingNodeIdx + 1];

                bool isHalo = false;
                for (int a = 0; a < assignment.nRanges(rank); ++a)
                {
                    I assignmentStart = assignment.rangeStart(rank, a);
                    I assignmentEnd   = assignment.rangeEnd(rank, a);

                    if (collidingNodeStart < assignmentStart || collidingNodeEnd > assignmentEnd)
                    {
                        // node with index collidingNodeIdx is a halo node
                        isHalo = true;
                    }
                }
                if (isHalo)
                {
                    // check if remote node +halo also overlaps with internal node
                    Box<int> remoteNodeBox = makeHaloBox(collidingNodeStart, collidingNodeEnd,
                                                         interactionRadii[collidingNodeIdx], box);
                    if (overlap(tree[nodeIdx], tree[nodeIdx+1], remoteNodeBox))
                    {
                        haloPairs.emplace_back(nodeIdx, collidingNodeIdx);
                    }
                }
            }
        }
    }
}

/*! \brief Compute send/receive node lists from halo pair node indices
 *
 * @tparam     I                32- or 64-bit unsigned integer
 * @param[in]  tree             cornerstone octree
 * @param[in]  assignment       stores which rank owns which part of the SFC
 * @param[in]  haloPairs        list of mutually overlapping pairs of local/remote nodes
 * @param[out] incomingNodes    sorted list of halo nodes to be received,
 *                              grouped by source rank
 * @param[out] outgoingNodes    sorted list of internal nodes to be sent,
 *                              grouped by destination rank
 */
template<class I>
void computeSendRecvNodeList(const std::vector<I>& tree,
                             const SpaceCurveAssignment<I>& assignment,
                             const std::vector<pair<int>>& haloPairs,
                             std::vector<std::vector<int>>& incomingNodes,
                             std::vector<std::vector<int>>& outgoingNodes)
{
    // needed to efficiently look up the assigned rank of a given octree node
    SfcLookupKey<I> sfcLookup(assignment);

    incomingNodes.resize(assignment.nRanks());
    outgoingNodes.resize(assignment.nRanks());

    for (auto& p : haloPairs)
    {
        // as defined in findHalos, the internal node index is stored first
        int internalNodeIdx = p[0];
        int remoteNodeIdx   = p[1];

        int remoteRank = sfcLookup.findRank(tree[remoteNodeIdx]);

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