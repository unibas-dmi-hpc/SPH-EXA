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

#include "cstone/domain/layout.hpp"
#include "cstone/focus/exchange_focus.hpp"
#include "cstone/focus/octree_focus.hpp"

namespace cstone
{

//! @brief A fully traversable octree with a local focus
template<class KeyType>
class FocusedOctree
{
public:
    /*! @brief constructor
     *
     * @param myRank        executing rank id
     * @param numRanks      number of ranks
     * @param bucketSize    Maximum number of particles per leaf inside the focus area
     * @param theta         Opening angle parameter for a min-distance MAC criterion.
     *                      In a converged FocusedOctree, each node outside the focus area
     *                      passes the min-distance MAC with theta as the parameter w.r.t
     *                      to any point inside the focus area.
     */
    FocusedOctree(int myRank, int numRanks, unsigned bucketSize, float theta)
        : myRank_(myRank)
        , numRanks_(numRanks)
        , theta_(theta)
        , treelets_(numRanks_)
        , tree_(bucketSize)
        , counts_{bucketSize + 1}
        , macs_{1}
    {
    }

    /*! @brief Update the tree structure according to previously calculated criteria (MAC and particle counts)
     *
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
     */
    bool updateTree(gsl::span<const int> peerRanks,
                    const SpaceCurveAssignment& assignment,
                    gsl::span<const KeyType> globalTreeLeaves)
    {
        if (rebalanceStatus_ != Criteria::valid)
        {
            throw std::runtime_error("Call to updateCriteria required before updating the tree structure\n");
        }

        KeyType focusStart = globalTreeLeaves[assignment.firstNodeIdx(myRank_)];
        KeyType focusEnd   = globalTreeLeaves[assignment.lastNodeIdx(myRank_)];

        std::vector<KeyType> peerBoundaries;
        for (int peer : peerRanks)
        {
            peerBoundaries.push_back(globalTreeLeaves[assignment.firstNodeIdx(peer)]);
            peerBoundaries.push_back(globalTreeLeaves[assignment.lastNodeIdx(peer)]);
        }

        bool converged   = tree_.update(focusStart, focusEnd, peerBoundaries, counts_, macs_);
        rebalanceStatus_ = Criteria::invalid;

        translateAssignment(assignment, globalTreeLeaves, treeLeaves(), peerRanks, myRank_, assignment_);

        return converged;
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
    void updateCriteria(const Box<T>& box,
                        gsl::span<const KeyType> particleKeys,
                        gsl::span<const int> peerRanks,
                        const SpaceCurveAssignment& assignment,
                        gsl::span<const KeyType> globalTreeLeaves,
                        gsl::span<const unsigned> globalCounts)
    {
        assert(std::is_sorted(particleKeys.begin(), particleKeys.end()));

        //! 1st regeneration step: local data
        updateMac(box, particleKeys, assignment, globalTreeLeaves);

        gsl::span<const KeyType> leaves = treeLeaves();
        leafCounts_.resize(nNodes(leaves));
        // local node counts
        computeNodeCounts(leaves.data(), leafCounts_.data(), nNodes(leaves), particleKeys.data(),
                          particleKeys.data() + particleKeys.size(), std::numeric_limits<unsigned>::max(), true);

        //! 2nd regeneration step: data from neighboring peers
        std::vector<MPI_Request> treeletRequests;
        exchangeTreelets(peerRanks, assignment_, leaves, treelets_, treeletRequests);
        exchangeTreeletCounts(peerRanks, treelets_, assignment_, particleKeys, leafCounts_, treeletRequests);
        MPI_Waitall(int(peerRanks.size()), treeletRequests.data(), MPI_STATUS_IGNORE);

        //! 3rd regeneration step: global data
        auto globalCountIndices = invertRanges(0, assignment_, nNodes(leaves));

        // particle counts for leaf nodes in treeLeaves() / leafCounts():
        //   Node indices [firstFocusNode:lastFocusNode] got assigned counts from local particles.
        //   Node index ranges listed in requestIndices got assigned counts from peer ranks.
        //   All remaining indices need to get their counts from the global tree.
        //   They are stored in globalCountIndices.

        for (auto ip : globalCountIndices)
        {
            countRequestParticles(globalTreeLeaves, globalCounts, leaves.subspan(ip.start(), ip.count() + 1),
                                  gsl::span<unsigned>(leafCounts_.data() + ip.start(), ip.count()));
        }

        counts_.resize(tree_.octree().numTreeNodes());
        upsweepSum<unsigned>(octree(), leafCounts_, counts_);

        rebalanceStatus_ = Criteria::valid;
    }

    //! @brief update the tree structure and regenerate the mac and counts criteria
    template<class T>
    bool update(const Box<T>& box,
                gsl::span<const KeyType> particleKeys,
                gsl::span<const int> peers,
                const SpaceCurveAssignment& assignment,
                gsl::span<const KeyType> globalTreeLeaves,
                gsl::span<const unsigned> globalCounts)
    {
        bool converged = updateTree(peers, assignment, globalTreeLeaves);
        updateCriteria(box, particleKeys, peers, assignment, globalTreeLeaves, globalCounts);
        return converged;
    }

    //! @brief update until converged with a simple min-distance MAC
    template<class T>
    void converge(const Box<T>& box,
                  gsl::span<const KeyType> particleKeys,
                  gsl::span<const int> peers,
                  const SpaceCurveAssignment& assignment,
                  gsl::span<const KeyType> globalTreeLeaves,
                  gsl::span<const unsigned> globalCounts)
    {
        int converged = 0;
        while (converged != numRanks_)
        {
            converged = update(box, particleKeys, peers, assignment, globalTreeLeaves, globalCounts);
            MPI_Allreduce(MPI_IN_PLACE, &converged, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        }
    }

    //! @brief the fully linked traversable octree
    const Octree<KeyType>& octree() const { return tree_.octree(); }
    //! @brief the cornerstone leaf cell array
    gsl::span<const KeyType> treeLeaves() const { return tree_.treeLeaves(); }
    //! @brief the assignment of the focus tree leaves to peer ranks
    gsl::span<const TreeIndexPair> assignment() const { return assignment_; }

    gsl::span<const unsigned> leafCounts() const
    {
        if (rebalanceStatus_ != Criteria::valid)
        {
            throw std::runtime_error("Tree structure is outdated, need to call updateCriteria\n");
        }
        return leafCounts_;
    }

private:
    template<class T>
    void updateMac(const Box<T>& box,
                   gsl::span<const KeyType> particleKeys,
                   const SpaceCurveAssignment& assignment,
                   gsl::span<const KeyType> globalTreeLeaves)
    {
        KeyType focusStart = globalTreeLeaves[assignment.firstNodeIdx(myRank_)];
        KeyType focusEnd   = globalTreeLeaves[assignment.lastNodeIdx(myRank_)];
        assert(particleKeys.front() >= focusStart && particleKeys.back() < focusEnd);

        macs_.resize(tree_.octree().numTreeNodes());
        markMac(tree_.octree(), box, focusStart, focusEnd, 1.0 / theta_, macs_.data());
    }

    enum class Criteria
    {
        valid,
        invalid
    };

    //! @brief the executing rank
    int myRank_;
    //! @brief the total number of ranks
    int numRanks_;
    //! @brief opening angle refinement criterion
    float theta_;

    //! @brief the tree structures that the peers have for the domain of the executing rank (myRank_)
    std::vector<std::vector<KeyType>> treelets_;

    FocusedOctreeCore<KeyType> tree_;

    //! @brief particle counts of the focused tree leaves
    std::vector<unsigned> leafCounts_;
    //! @brief particle counts of the full tree, including internal nodes
    std::vector<unsigned> counts_;
    //! @brief mac evaluation result relative to focus area (pass or fail)
    std::vector<char> macs_;

    //! @brief the assignment of peer ranks to tree_.treeLeaves()
    std::vector<TreeIndexPair> assignment_;

    //! @brief the status of the macs_ and counts_ rebalance criteria
    Criteria rebalanceStatus_{Criteria::valid};
};

} // namespace cstone
