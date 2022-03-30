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
        peers_.resize(peerRanks.size());
        std::copy(peerRanks.begin(), peerRanks.end(), peers_.begin());

        KeyType focusStart = globalTreeLeaves[assignment.firstNodeIdx(myRank_)];
        KeyType focusEnd   = globalTreeLeaves[assignment.lastNodeIdx(myRank_)];
        // init on first call
        if (prevFocusStart == 0 && prevFocusEnd == 0)
        {
            prevFocusStart = focusStart;
            prevFocusEnd   = focusEnd;
        }

        std::vector<KeyType> enforcedKeys;
        enforcedKeys.reserve(peers_.size() * 2);

        focusTransfer(treeLeaves(), myRank_, prevFocusStart, prevFocusEnd, focusStart, focusEnd, enforcedKeys);
        for (int peer : peers_)
        {
            enforcedKeys.push_back(globalTreeLeaves[assignment.firstNodeIdx(peer)]);
            enforcedKeys.push_back(globalTreeLeaves[assignment.lastNodeIdx(peer)]);
        }

        bool converged = tree_.update(focusStart, focusEnd, enforcedKeys, counts_, macs_);
        translateAssignment(assignment, globalTreeLeaves, treeLeaves(), peers_, myRank_, assignment_);

        prevFocusStart   = focusStart;
        prevFocusEnd     = focusEnd;
        rebalanceStatus_ = invalid;
        return converged;
    }

    /*! @brief Perform a global update of the tree structure
     *
     * @param[in] box              global coordinate bounding box
     * @param[in] particleKeys     SFC keys of local particles
     * @param[in] myRank           ID of the executing rank
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
        exchangeTreelets(peers_, assignment_, leaves, treelets_, treeletRequests);
        exchangeTreeletCounts(peers_, treelets_, assignment_, leaves, leafCounts_, treeletRequests);
        MPI_Waitall(int(peers_.size()), treeletRequests.data(), MPI_STATUS_IGNORE);

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
        scatter(octree().internalOrder(), leafCounts_.data(), counts_.data());
        upsweep(octree().levelRange(), octree().childOffsets(), counts_.data(), SumCombination<unsigned>{});

        rebalanceStatus_ |= countsCriterion;
    }

    template<class T>
    void peerExchange(gsl::span<T> quantities, int commTag) const
    {
        exchangeTreeletGeneral<T>(peers_, treelets_, assignment_, octree().nodeKeys(), octree().levelRange(),
                                  octree().internalOrder(), quantities, commTag);
    }

    /*! @brief transfer quantities of leaf cells inside the focus into a global array
     *
     * @tparam T
     * @param[in]  globalLeaves      cstone SFC key leaf cell array of the global tree
     * @param[in]  localQuantities   cell properties of the locally focused tree
     * @param[out] globalQuantities  cell properties of the global tree
     */
    template<class T>
    void populateGlobal(gsl::span<const KeyType> globalLeaves,
                        gsl::span<const T> localQuantities,
                        gsl::span<T> globalQuantities) const
    {
        TreeNodeIndex firstGlobalIdx = findNodeAbove(globalLeaves, prevFocusStart);
        TreeNodeIndex lastGlobalIdx  = findNodeAbove(globalLeaves, prevFocusEnd);
        // make sure that the focus is resolved exactly in the global tree
        assert(globalLeaves[firstIdx] == prevFocusStart);
        assert(globalLeaves[lastIdx] == prevFocusEnd);

#pragma omp parallel for schedule(static)
        for (TreeNodeIndex globalIdx = firstGlobalIdx; globalIdx < lastGlobalIdx; ++globalIdx)
        {
            TreeNodeIndex localIdx      = octree().locate(globalLeaves[globalIdx], globalLeaves[globalIdx + 1]);
            globalQuantities[globalIdx] = localQuantities[localIdx];
            assert(octree().codeStart(localIdx) == globalLeaves[globalIdx]);
            assert(octree().codeEnd(localIdx) == globalLeaves[globalIdx + 1]);
        }
    }

    /*! @brief transfer missing cell quantities from global tree into localQuantities
     *
     * @tparam T
     * @param[in]  globalTree
     * @param[in]  globalQuantities   tree cell properties for each cell in @p globalTree include internal cells
     * @param[out] localQuantities    local tree cell properties
     */
    template<class T>
    void extractGlobal(const Octree<KeyType>& globalTree,
                       gsl::span<const T> globalQuantities,
                       gsl::span<T> localQuantities) const
    {
        gsl::span<const KeyType> localLeaves = treeLeaves();
        //! requestIndices: range of leaf cell indices in the locally focused tree that need global information
        auto requestIndices = invertRanges(0, assignment_, octree().numLeafNodes());
        for (auto range : requestIndices)
        {
            //! from global tree, pull in missing elements into locally focused tree
            for (TreeNodeIndex i = range.start(); i < range.end(); ++i)
            {
                TreeNodeIndex globalIndex    = globalTree.locate(localLeaves[i], localLeaves[i + 1]);
                TreeNodeIndex internalIdx    = octree().toInternal(i);
                localQuantities[internalIdx] = globalQuantities[globalIndex];
            }
        }
    }

    template<class T, class Tm>
    void updateCenters(gsl::span<const T> x,
                       gsl::span<const T> y,
                       gsl::span<const T> z,
                       gsl::span<const Tm> m,
                       const SpaceCurveAssignment& assignment,
                       const Octree<KeyType>& globalTree,
                       const Box<T>& box)
    {
        //! compute temporary pre-halo exchange particle layout for local particles only
        std::vector<LocalIndex> layout(leafCounts_.size() + 1, 0);
        TreeNodeIndex firstIdx = assignment_[myRank_].start();
        TreeNodeIndex lastIdx  = assignment_[myRank_].end();
        std::exclusive_scan(leafCounts_.begin() + firstIdx, leafCounts_.begin() + lastIdx + 1,
                            layout.begin() + firstIdx, 0);

        globalCenters_.resize(globalTree.numTreeNodes());
        centers_.resize(octree().numTreeNodes());

#pragma omp parallel for schedule(static)
        for (TreeNodeIndex leafIdx = 0; leafIdx < octree().numLeafNodes(); ++leafIdx)
        {
            //! prepare local leaf centers
            TreeNodeIndex nodeIdx = octree().toInternal(leafIdx);
            centers_[nodeIdx] =
                massCenter<RealType>(x.data(), y.data(), z.data(), m.data(), layout[leafIdx], layout[leafIdx + 1]);
        }

        //! upsweep with local data in place
        upsweep(octree().levelRange(), octree().childOffsets(), centers_.data(), CombineSourceCenter<T>{});
        //! exchange information with peer close to focus
        peerExchange<SourceCenterType<T>>(centers_, static_cast<int>(P2pTags::focusPeerCenters));
        //! global exchange for the top nodes that are bigger than local domains
        std::vector<SourceCenterType<T>> globalLeafCenters(globalTree.numLeafNodes());
        populateGlobal<SourceCenterType<T>>(globalTree.treeLeaves(), centers_, globalLeafCenters);
        mpiAllreduce(MPI_IN_PLACE, globalLeafCenters.data(), globalLeafCenters.size(), MPI_SUM);
        scatter(globalTree.internalOrder(), globalLeafCenters.data(), globalCenters_.data());
        upsweep(globalTree.levelRange(), globalTree.childOffsets(), globalCenters_.data(), CombineSourceCenter<T>{});
        extractGlobal<SourceCenterType<T>>(globalTree, globalCenters_, centers_);

        //! upsweep with all (leaf) data in place
        upsweep(octree().levelRange(), octree().childOffsets(), centers_.data(), CombineSourceCenter<T>{});
        //! calculate mac radius for each cell based on location of expansion centers
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
    void
    updateMinMac(const Box<T>& box, const SpaceCurveAssignment& assignment, gsl::span<const KeyType> globalTreeLeaves)
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
    void
    updateVecMac(const Box<T>& box, const SpaceCurveAssignment& assignment, gsl::span<const KeyType> globalTreeLeaves)
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
        updateCounts(particleKeys, globalTreeLeaves, globalCounts);
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
    //! @brief Expansion (com) centers of each global cell
    gsl::span<const SourceCenterType<RealType>> globalExpansionCenters() const { return globalCenters_; }
    //! @brief access multipole acceptance status of each cell
    gsl::span<const char> macs() const { return macs_; }
    //! brief particle counts per focus tree leaf cell
    gsl::span<const unsigned> leafCounts() const { return leafCounts_; }

    void addMacs(gsl::span<int> haloFlags) const
    {
#pragma omp parallel for schedule(static)
        for (TreeNodeIndex i = 0; i < haloFlags.ssize(); ++i)
        {
            size_t iIdx = octree().toInternal(i);
            if (macs_[iIdx] && !haloFlags[i]) { haloFlags[i] = 1; }
        }
    }

private:
    enum Status : int
    {
        invalid         = 0,
        countsCriterion = 1,
        macCriterion    = 2,
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

    //! @brief list of peer ranks from last call to updateTree()
    std::vector<int> peers_;
    //! @brief the tree structures that the peers have for the domain of the executing rank (myRank_)
    std::vector<std::vector<KeyType>> treelets_;

    FocusedOctreeCore<KeyType> tree_;

    //! @brief previous iteration focus start
    KeyType prevFocusStart = 0;
    //! @brief previous iteration focus end
    KeyType prevFocusEnd = 0;

    //! @brief particle counts of the focused tree leaves, tree_.treeLeaves()
    std::vector<unsigned> leafCounts_;
    //! @brief particle counts of the full tree, tree_.octree()
    std::vector<unsigned> counts_;
    //! @brief mac evaluation result relative to focus area (pass or fail)
    std::vector<char> macs_;
    //! @brief the expansion (com) centers of each cell of tree_.octree()
    std::vector<SourceCenterType<RealType>> centers_;
    //! @brief we also need to hold on to the expansion centers of the global tree for the multipole upsweep
    std::vector<SourceCenterType<RealType>> globalCenters_;
    //! @brief the assignment of peer ranks to tree_.treeLeaves()
    std::vector<TreeIndexPair> assignment_;

    //! @brief the status of the macs_ and counts_ rebalance criteria
    int rebalanceStatus_{valid};
};

/*! @brief exchange data of non-peer (beyond focus) tree cells
 *
 * @tparam        T                an arithmetic type, or compile-time fix-sized arrays thereof
 * @tparam        F                function object for octree upsweep
 * @param[in]     globalTree       the same global (replicated on all ranks) tree that was used for peer rank
 *                                 detection
 * @param[out]    globalQuantities an array of length @p globalTree.numTreeNodes(), will be populated with
 *                                 information from @p quantities
 * @param[inout]  quantities       an array of length octree().numTreeNodes() with cell properties of the
 *                                 locally focused octree
 *
 * The data flow is:
 *  local cells of quantities -> leaves of globalQuantities -> global collective communication -> upsweep
 *   -> back-contribution of global cell quantities into the
 *
 * Precondition:  quantities contains valid data for each cell, including internal cells,
 *                that falls into the focus range of the executing
 *                rank
 * Postcondition: each element of quantities corresponding to cells non-local and not owned by any of the peer
 *                ranks contains data obtained through global collective communication between ranks
 */
template<class MType, class T, class KeyType, class F>
void globalMultipoleExchange(const Octree<KeyType>& globalOctree,
                             const FocusedOctree<KeyType, T>& focusTree,
                             gsl::span<const SourceCenterType<T>> globalCenters,
                             gsl::span<MType> multipoles,
                             F&& upsweepFunction)
{
    TreeNodeIndex numGlobalLeaves = globalOctree.numLeafNodes();
    std::vector<MType> globalLeafMultipoles(numGlobalLeaves);
    focusTree.template populateGlobal<MType>(globalOctree.treeLeaves(), multipoles, globalLeafMultipoles);

    //! exchange global leaves
    mpiAllreduce(MPI_IN_PLACE, globalLeafMultipoles.data(), numGlobalLeaves, MPI_SUM);

    std::vector<MType> globalMultipoles(globalOctree.numTreeNodes());
    scatter(globalOctree.internalOrder(), globalLeafMultipoles.data(), globalMultipoles.data());
    //! upsweep with the global tree
    upsweepFunction(globalOctree.levelRange(), globalOctree.childOffsets(), globalCenters.data(),
                    globalMultipoles.data());

    focusTree.template extractGlobal<MType>(globalOctree, globalMultipoles, multipoles);
}

} // namespace cstone
