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
#include "cstone/focus/source_center.hpp"

namespace cstone
{

//! @brief A fully traversable octree with a local focus
template<class KeyType, class RealType>
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
        if (rebalanceStatus_ != valid)
        {
            throw std::runtime_error("update of criteria required before updating the tree structure\n");
        }

        KeyType focusStart = globalTreeLeaves[assignment.firstNodeIdx(myRank_)];
        KeyType focusEnd   = globalTreeLeaves[assignment.lastNodeIdx(myRank_)];
        // init on first call
        if (prevFocusStart == 0 && prevFocusEnd == 0)
        {
            prevFocusStart = focusStart;
            prevFocusEnd   = focusEnd;
        }

        std::vector<KeyType> enforcedKeys;
        enforcedKeys.reserve(peerRanks.size() * 2);

        focusTransfer(treeLeaves(), myRank_, prevFocusStart, prevFocusEnd, focusStart, focusEnd, enforcedKeys);
        for (int peer : peerRanks)
        {
            enforcedKeys.push_back(globalTreeLeaves[assignment.firstNodeIdx(peer)]);
            enforcedKeys.push_back(globalTreeLeaves[assignment.lastNodeIdx(peer)]);
        }

        bool converged = tree_.update(focusStart, focusEnd, enforcedKeys, counts_, macs_);
        translateAssignment(assignment, globalTreeLeaves, treeLeaves(), peerRanks, myRank_, assignment_);

        prevFocusStart = focusStart;
        prevFocusEnd   = focusEnd;
        rebalanceStatus_ = invalid;
        return converged;
    }

    /*! @brief Perform a global update of the tree structure
     *
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
    void updateCounts(gsl::span<const KeyType> particleKeys,
                      gsl::span<const int> peerRanks,
                      const SpaceCurveAssignment& assignment,
                      gsl::span<const KeyType> globalTreeLeaves,
                      gsl::span<const unsigned> globalCounts)
    {
        assert(std::is_sorted(particleKeys.begin(), particleKeys.end()));

        gsl::span<const KeyType> leaves = treeLeaves();
        leafCounts_.resize(nNodes(leaves));

        // local node counts
        computeNodeCounts(leaves.data(), leafCounts_.data(), nNodes(leaves), particleKeys.data(),
                          particleKeys.data() + particleKeys.size(), std::numeric_limits<unsigned>::max(), true);

        // counts from neighboring peers
        std::vector<MPI_Request> treeletRequests;
        exchangeTreelets(peerRanks, assignment_, leaves, treelets_, treeletRequests);
        exchangeTreeletCounts(peerRanks, treelets_, assignment_, leaves, leafCounts_, treeletRequests);
        MPI_Waitall(int(peerRanks.size()), treeletRequests.data(), MPI_STATUS_IGNORE);

        // global counts
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

        rebalanceStatus_ |= countsCriterion;
    }

    template<class T>
    void peerExchange(gsl::span<const int> peerRanks,
                      gsl::span<T> quantities,
                      int commTag)
    {
        exchangeTreeletGeneral<T>(peerRanks, treelets_, assignment_, octree().nodeKeys(), octree().levelRange(),
                                  octree().internalOrder(), quantities, commTag);
    }

    /*! @brief exchange data of non-peer (beyond focus) tree cells
     *
     * @tparam        T            an arithmetic type, or compile-time fix-sized arrays thereof
     * @tparam        F            function object for octree upsweep
     * @param[in]     globalTree   the same global (replicated on all ranks) tree that was used for peer rank detection
     * @param[inout]  quantities   an array of length octree().numTreeNodes()
     *
     * Precondition:  quantities contains valid data for each cell, including internal cells,
     *                that falls into the focus range of the executing
     *                rank
     * Postcondition: each element of quantities corresponding to cells non-local and not owned by any of the peer
     *                ranks contains data obtained through global collective communication between ranks
     */
    template<class T, class F>
    void globalExchange(const Octree<KeyType>& globalTree,
                        T* quantities,
                        F&& upsweepFunction)
    {
        TreeNodeIndex numGlobalLeaves = globalTree.numLeafNodes();
        std::vector<T> globalLeafQuantities(numGlobalLeaves);
        // fetch local quantities into globalLeaves
        gsl::span<const KeyType> globalLeaves = globalTree.treeLeaves();

        TreeNodeIndex firstIdx = findNodeAbove(globalLeaves, prevFocusStart);
        TreeNodeIndex lastIdx  = findNodeAbove(globalLeaves, prevFocusEnd);
        assert(globalLeaves[firstIdx] == prevFocusStart);
        assert(globalLeaves[lastIdx] == prevFocusEnd);

        #pragma omp parallel for schedule(static)
        for (TreeNodeIndex globalIdx = firstIdx; globalIdx < lastIdx; ++globalIdx)
        {
            TreeNodeIndex localIdx          = octree().locate(globalLeaves[globalIdx], globalLeaves[globalIdx + 1]);
            globalLeafQuantities[globalIdx] = quantities[localIdx];
            assert(octree().codeStart(localIdx) == globalLeaves[globalIdx]);
            assert(octree().codeEnd(localIdx) == globalLeaves[globalIdx + 1]);
        }
        mpiAllreduce(MPI_IN_PLACE, globalLeafQuantities.data(), numGlobalLeaves, MPI_SUM);

        std::vector<T> globalQuantities(globalTree.numTreeNodes());
        upsweep(globalTree, globalLeafQuantities.data(), globalQuantities.data(), std::forward<F>(upsweepFunction));

        gsl::span<const KeyType> localLeaves = treeLeaves();
        auto globalIndices = invertRanges(0, assignment_, octree().numLeafNodes());
        for (auto range : globalIndices)
        {
            for (TreeNodeIndex i = range.start(); i < range.end(); ++i)
            {
                TreeNodeIndex globalIndex = globalTree.locate(localLeaves[i], localLeaves[i + 1]);
                TreeNodeIndex internalIdx = octree().toInternal(i);
                quantities[internalIdx]   = globalQuantities[globalIndex];
            }
        }
    }

    template<class T, class Tm>
    void updateCenters(gsl::span<const T> x,
                       gsl::span<const T> y,
                       gsl::span<const T> z,
                       gsl::span<const Tm> m,
                       gsl::span<const int> peerRanks,
                       const SpaceCurveAssignment& assignment,
                       const Octree<KeyType>& globalTree,
                       const Box<T>& box)
    {
        // compute temporary pre-halo exchange particle layout for local particles only
        std::vector<LocalIndex> layout(leafCounts_.size() + 1, 0);
        TreeNodeIndex firstIdx = assignment_[myRank_].start();
        TreeNodeIndex lastIdx  = assignment_[myRank_].end();
        std::exclusive_scan(leafCounts_.begin() + firstIdx, leafCounts_.begin() + lastIdx + 1,
                            layout.begin() + firstIdx, 0);

        centers_.resize(octree().numTreeNodes());

        #pragma omp parallel for schedule(static)
        for (TreeNodeIndex leafIdx = 0; leafIdx < octree().numLeafNodes(); ++leafIdx)
        {
            TreeNodeIndex nodeIdx = octree().toInternal(leafIdx);
            centers_[nodeIdx] =
                massCenter<RealType>(x.data(), y.data(), z.data(), m.data(), layout[leafIdx], layout[leafIdx + 1]);
        }

        upsweep(octree(), centers_.data(), CombineSourceCenter<T>{});
        peerExchange<SourceCenterType<T>>(peerRanks, centers_, static_cast<int>(P2pTags::focusPeerCenters));
        globalExchange(globalTree, centers_.data(), CombineSourceCenter<T>{});
        upsweep(octree(), centers_.data(), CombineSourceCenter<T>{});

        setMac<T>(octree().nodeKeys(), centers_, 1.0 / theta_, box);
    }

    /*! @brief Update the MAC criteria based on a min distance MAC
     *
     * @tparam    T                float or double
     * @param[in] box              global coordinate bounding box
     * @param[in] assignment       assignment of the global leaf tree to ranks
     * @param[in] globalTreeLeaves global cornerstone leaf tree
     */
    template<class T>
    void updateMinMac(const Box<T>& box,
                      const SpaceCurveAssignment& assignment,
                      gsl::span<const KeyType> globalTreeLeaves)
    {
        KeyType focusStart = globalTreeLeaves[assignment.firstNodeIdx(myRank_)];
        KeyType focusEnd   = globalTreeLeaves[assignment.lastNodeIdx(myRank_)];

        macs_.resize(tree_.octree().numTreeNodes());
        markMac(tree_.octree(), box, focusStart, focusEnd, 1.0 / theta_, macs_.data());

        rebalanceStatus_ |= macCriterion;
    }

    /*! @brief Update the MAC criteria based on the vector MAC
     *
     * @tparam    T                float or double
     * @param[in] box              global coordinate bounding box
     * @param[in] assignment       assignment of the global leaf tree to ranks
     * @param[in] globalTreeLeaves global cornerstone leaf tree
     */
    template<class T>
    void updateVecMac(const Box<T>& box,
                      const SpaceCurveAssignment& assignment,
                      gsl::span<const KeyType> globalTreeLeaves)
    {
        KeyType focusStart = globalTreeLeaves[assignment.firstNodeIdx(myRank_)];
        KeyType focusEnd   = globalTreeLeaves[assignment.lastNodeIdx(myRank_)];

        macs_.resize(octree().numTreeNodes());
        markVecMac(octree(), centers_.data(), box, focusStart, focusEnd, macs_.data());

        rebalanceStatus_ |= macCriterion;
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
        updateCounts(particleKeys, peers, assignment, globalTreeLeaves, globalCounts);
        updateMinMac(box, assignment, globalTreeLeaves);
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
    //! @brief Expansion (com) centers of each cell
    gsl::span<const SourceCenterType<RealType>> expansionCenters() const { return centers_; }
    //! @brief access multipole acceptance status of each cell
    gsl::span<const char> macs() const { return macs_; }
    //! brief particle counts per focus tree leaf cell
    gsl::span<const unsigned> leafCounts() const { return leafCounts_; }

    void addMacs(gsl::span<int> haloFlags) const
    {
        #pragma omp parallel for schedule(static)
        for (TreeNodeIndex i = 0; i < haloFlags.size(); ++i)
        {
            size_t iIdx = octree().toInternal(i);
            if (macs_[iIdx] && !haloFlags[i])
            {
                haloFlags[i] = 1;
            }
        }
    }

private:

    enum Status : int
    {
        invalid = 0,
        countsCriterion = 1,
        macCriterion = 2,
        // the status is valid for rebalancing if both the counts and macs have been updated
        // since the last call to updateTree
        valid = countsCriterion | macCriterion
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

    //! @brief previous iteration focus start
    KeyType prevFocusStart = 0;
    //! @brief previous iteration focus end
    KeyType prevFocusEnd   = 0;

    //! @brief particle counts of the focused tree leaves
    std::vector<unsigned> leafCounts_;
    //! @brief particle counts of the full tree, including internal nodes
    std::vector<unsigned> counts_;
    //! @brief mac evaluation result relative to focus area (pass or fail)
    std::vector<char> macs_;
    //! @brief the expansion (com) centers of each treeLeaves cell
    std::vector<SourceCenterType<RealType>> centers_;

    //! @brief the assignment of peer ranks to tree_.treeLeaves()
    std::vector<TreeIndexPair> assignment_;

    //! @brief the status of the macs_ and counts_ rebalance criteria
    int rebalanceStatus_{valid};
};

} // namespace cstone
