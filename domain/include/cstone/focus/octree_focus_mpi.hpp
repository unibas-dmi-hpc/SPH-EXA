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
 * @brief Traits and functors for the MPI-enabled FocusedOctree
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/focus/exchange_focus.hpp"
#include "cstone/focus/octree_focus.hpp"
#include "cstone/focus/request_indices.hpp"

namespace cstone
{

//! @brief A fully traversable octree with a local focus
template<class KeyType>
class FocusedOctree
{
public:

    /*! @brief constructor
     *
     * @param bucketSize    Maximum number of particles per leaf inside the focus area
     * @param theta         Opening angle parameter for a min-distance MAC criterion.
     *                      In a converged FocusedOctree, each node outside the focus area
     *                      passes the min-distance MAC with theta as the parameter w.r.t
     *                      to any point inside the focus area.
     */
    FocusedOctree(unsigned bucketSize, float theta)
        : theta_(theta), tree_(bucketSize), counts_{bucketSize + 1}, macs_{1}
    {
    }

    /*! @brief Perform a global update of the tree structure
     *
     * @tparam    T                float or double
     * @param[in] box              global coordinate bounding box
     * @param[in] particleKeys     SFC keys of local particles
     * @param[in] myRank           ID of the executing rank
     * @param[in] peerRanks        list of ranks that have nodes that fail the MAC criterion
     *                             w.r.t to the assigned SFC part of @p myRank
     *                             use e.g. findPeersMac to calculate this list
     * @param[in] assignment       assignment of the global leaf tree to ranks
     * @param[in] globalTreeLeaves global cornerstone leaf tree
     * @param[in] globalCounts     global cornerstone leaf tree counts
     * @return                     true if the tree structure did not change
     *
     * The part of the SFC that is assigned to @p myRank is considered as the focus area.
     *
     * Preconditions:
     *  - The provided assignment and globalTreeLeaves are the same as what was used for
     *    calculating the list of peer ranks with findPeersMac. (not checked)
     *  - All local particle keys must lie within the assignment of @p myRank (checked)
     *    and must be sorted in ascending order (checked)
     */
    template<class T>
    bool update(const Box<T>& box, gsl::span<const KeyType> particleKeys, int myRank, gsl::span<const int> peerRanks,
                const SpaceCurveAssignment& assignment, gsl::span<const KeyType> globalTreeLeaves,
                gsl::span<const unsigned> globalCounts)
    {
        KeyType focusStart = globalTreeLeaves[assignment.firstNodeIdx(myRank)];
        KeyType focusEnd   = globalTreeLeaves[assignment.lastNodeIdx(myRank)];

        assert(particleKeys.front() >= focusStart && particleKeys.back() < focusEnd);

        std::vector<KeyType> peerBoundaries;
        for (int peer : peerRanks)
        {
            peerBoundaries.push_back(globalTreeLeaves[assignment.firstNodeIdx(peer)]);
            peerBoundaries.push_back(globalTreeLeaves[assignment.lastNodeIdx(peer)]);
        }

        bool converged = tree_.update(particleKeys, focusStart, focusEnd, peerBoundaries, counts_, macs_);

        gsl::span<const KeyType> leaves = treeLeaves();

        //! 1st regeneration step: local data

        macs_.resize(tree_.octree().numTreeNodes());
        markMac(tree_.octree(), box, focusStart, focusEnd, 1.0 / theta_, macs_.data());

        counts_.resize(nNodes(leaves));
        // local node counts
        computeNodeCounts(leaves.data(), counts_.data(), nNodes(leaves), particleKeys.data(),
                          particleKeys.data() + particleKeys.size(), std::numeric_limits<unsigned>::max(), true);

        //! 2nd regeneration step: data from neighboring peers

        auto requestIndices = findRequestIndices(peerRanks, assignment, globalTreeLeaves, leaves);

        exchangePeerCounts(peerRanks, requestIndices, particleKeys, leaves, counts_);

        //! 3rd regeneration step: global data

        TreeNodeIndex firstFocusNode = findNodeAbove(leaves, focusStart);
        TreeNodeIndex lastFocusNode  = findNodeBelow(leaves, focusEnd);
        requestIndices.emplace_back(firstFocusNode, lastFocusNode);
        std::sort(requestIndices.begin(), requestIndices.end());
        auto globalCountIndices = invertRanges(0, requestIndices, nNodes(leaves));

        // particle counts for leaf nodes in treeLeaves() / leafCounts():
        //   Node indices [firstFocusNode:lastFocusNode] got assigned counts from local particles.
        //   Node index ranges listed in requestIndices got assigned counts from peer ranks.
        //   All remaining indices need to get their counts from the global tree.
        //   They are stored in globalCountIndices.

        for (auto ip : globalCountIndices)
        {
            countRequestParticles(globalTreeLeaves, globalCounts, leaves.subspan(ip.start(), ip.count() + 1),
                                  gsl::span<unsigned>(counts_.data() + ip.start(), ip.count()));
        }

        return converged;
    }

    //! @brief like update, but repeat until converged on first call
    template<class T>
    void initAndUpdate(const Box<T>& box,
                       gsl::span<const KeyType> particleKeys,
                       int myRank,
                       gsl::span<const int> peers,
                       const SpaceCurveAssignment& assignment,
                       gsl::span<const KeyType> globalTreeLeaves,
                       gsl::span<const unsigned> globalCounts)
    {
        update(box, particleKeys, myRank, peers, assignment, globalTreeLeaves, globalCounts);

        if (firstCall_)
        {
            int numRanks;
            MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
            // we must not call updateGlobal again before all ranks have completed the previous call,
            // otherwise point-2-point messages from different updateGlobal calls can get mixed up
            MPI_Barrier(MPI_COMM_WORLD);
            int converged = 0;
            while (converged != numRanks)
            {
                converged = update(box, particleKeys, myRank, peers, assignment, globalTreeLeaves, globalCounts);
                MPI_Allreduce(MPI_IN_PLACE, &converged, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            }
            firstCall_ = false;
        }
    }

    const Octree<KeyType>& octree() const { return tree_.octree(); }

    gsl::span<const KeyType>  treeLeaves() const { return tree_.treeLeaves(); }
    gsl::span<const unsigned> leafCounts() const { return counts_; }

private:

    //! @brief opening angle refinement criterion
    float theta_;

    FocusedOctreeCore<KeyType> tree_;

    //! @brief particle counts of the focused tree leaves
    std::vector<unsigned> counts_;
    //! @brief mac evaluation result relative to focus area (pass or fail)
    std::vector<char> macs_;

    bool firstCall_{true};
};

} // namespace cstone
