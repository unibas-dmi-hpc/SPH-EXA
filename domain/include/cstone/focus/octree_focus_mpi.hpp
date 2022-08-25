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

#include <numeric>

#include "cstone/cuda/cuda_utils.hpp"
#include "cstone/domain/layout.hpp"
#include "cstone/focus/exchange_focus.hpp"
#include "cstone/focus/octree_focus.hpp"
#include "cstone/focus/source_center.hpp"
#include "cstone/focus/source_center_gpu.h"

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
        , bucketSize_(bucketSize)
        , treelets_(numRanks_)
        , counts_{bucketSize + 1}
        , macs_{1}
    {
        std::vector<KeyType> init{0, nodeRange<KeyType>(0)};
        tree_.update(init.data(), nNodes(init));
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

        focusTransfer(treeLeaves(), leafCounts(), bucketSize_, myRank_, prevFocusStart, prevFocusEnd, focusStart,
                      focusEnd, enforcedKeys);
        for (int peer : peers_)
        {
            enforcedKeys.push_back(globalTreeLeaves[assignment.firstNodeIdx(peer)]);
            enforcedKeys.push_back(globalTreeLeaves[assignment.lastNodeIdx(peer)]);
        }

        bool converged = CombinedUpdate<KeyType>::updateFocus(tree_, bucketSize_, focusStart, focusEnd, enforcedKeys,
                                                              counts_, macs_);
        translateAssignment(assignment, globalTreeLeaves, treeLeaves(), peers_, myRank_, assignment_);

        prevFocusStart   = focusStart;
        prevFocusEnd     = focusEnd;
        rebalanceStatus_ = invalid;
        return converged;
    }

    /*! @brief Perform a global update of the tree structure
     *
     * @param[in] particleKeys     SFC keys of local particles
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
    template<class DeviceVector = std::vector<KeyType>>
    void updateCounts(gsl::span<const KeyType> particleKeys,
                      gsl::span<const KeyType> globalTreeLeaves,
                      gsl::span<const unsigned> globalCounts,
                      DeviceVector&& scratch = std::vector<KeyType>{})
    {
        gsl::span<const KeyType> leaves = treeLeaves();
        leafCounts_.resize(nNodes(leaves));

        if constexpr (IsDeviceVector<std::decay_t<DeviceVector>>{})
        {
            TreeNodeIndex numNodes = tree_.numLeafNodes();

            size_t bytesTree  = round_up((numNodes + 1) * sizeof(KeyType), 128);
            size_t bytesCount = numNodes * sizeof(unsigned);
            size_t origSize   = reallocateDeviceBytes(scratch, bytesTree + bytesCount);
            auto* d_csTree    = reinterpret_cast<KeyType*>(rawPtr(scratch));
            auto* d_counts    = reinterpret_cast<unsigned*>(rawPtr(scratch)) + bytesTree / sizeof(unsigned);

            memcpyH2D(leaves.data(), leaves.size(), d_csTree);
            computeNodeCountsGpu(d_csTree, d_counts, numNodes, particleKeys.begin(), particleKeys.end(),
                                 std::numeric_limits<unsigned>::max(), false);
            memcpyD2H(d_counts, numNodes, leafCounts_.data());

            reallocateDevice(scratch, origSize, 1.0);
        }
        else
        {
            // local node counts
            assert(std::is_sorted(particleKeys.begin(), particleKeys.end()));
            computeNodeCounts(leaves.data(), leafCounts_.data(), nNodes(leaves), particleKeys.data(),
                              particleKeys.data() + particleKeys.size(), std::numeric_limits<unsigned>::max(), true);
        }

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

        counts_.resize(tree_.numTreeNodes());
        scatter(tree_.internalOrder(), leafCounts_.data(), counts_.data());
        upsweep(tree_.levelRange(), tree_.childOffsets(), counts_.data(), SumCombination<unsigned>{});

        rebalanceStatus_ |= countsCriterion;
    }

    template<class T>
    void peerExchange(gsl::span<T> quantities, int commTag) const
    {
        exchangeTreeletGeneral<T>(peers_, treelets_, assignment_, tree_.nodeKeys(), tree_.levelRange(),
                                  tree_.internalOrder(), quantities, commTag);
    }

    /*! @brief transfer quantities of leaf cells inside the focus into a global array
     *
     * @tparam     T                 an arithmetic type or compile-time constant size arrays thereof
     * @param[in]  globalLeaves      cstone SFC key leaf cell array of the global tree
     * @param[in]  localQuantities   cell properties of the locally focused tree, length = octree().numTreeNodes()
     * @param[out] globalQuantities  cell properties of the global tree
     */
    template<class T>
    void populateGlobal(gsl::span<const KeyType> globalLeaves,
                        gsl::span<const T> localQuantities,
                        gsl::span<T> globalQuantities) const
    {
        assert(localQuantities.size() == octree().numTreeNodes());

        TreeNodeIndex firstGlobalIdx = findNodeAbove(globalLeaves, prevFocusStart);
        TreeNodeIndex lastGlobalIdx  = findNodeAbove(globalLeaves, prevFocusEnd);
        // make sure that the focus is resolved exactly in the global tree
        assert(globalLeaves[firstGlobalIdx] == prevFocusStart);
        assert(globalLeaves[lastGlobalIdx] == prevFocusEnd);

#pragma omp parallel for schedule(static)
        for (TreeNodeIndex globalIdx = firstGlobalIdx; globalIdx < lastGlobalIdx; ++globalIdx)
        {
            TreeNodeIndex localIdx = tree_.locate(globalLeaves[globalIdx], globalLeaves[globalIdx + 1]);
            if (localIdx == tree_.numTreeNodes())
            {
                // If the global tree is fully converged, but the locally focused tree is just being built up
                // for the first time, it's possible that the global tree has a higher resolution than
                // the focused tree.
                continue;
            }
            assert(octree().codeStart(localIdx) == globalLeaves[globalIdx]);
            assert(octree().codeEnd(localIdx) == globalLeaves[globalIdx + 1]);
            globalQuantities[globalIdx] = localQuantities[localIdx];
        }
    }

    /*! @brief transfer missing cell quantities from global tree into localQuantities
     *
     * @tparam     T                 an arithmetic type or compile-time constant size arrays thereof
     * @param[in]  globalTree
     * @param[in]  globalQuantities  tree cell properties for each cell in @p globalTree include internal cells
     * @param[out] localQuantities   local tree cell properties
     */
    template<class T>
    void extractGlobal(const Octree<KeyType>& globalTree,
                       gsl::span<const T> globalQuantities,
                       gsl::span<T> localQuantities) const
    {
        gsl::span<const KeyType> localLeaves = treeLeaves();
        //! requestIndices: range of leaf cell indices in the locally focused tree that need global information
        auto requestIndices = invertRanges(0, assignment_, tree_.numLeafNodes());
        for (auto range : requestIndices)
        {
            //! from global tree, pull in missing elements into locally focused tree
            for (TreeNodeIndex i = range.start(); i < range.end(); ++i)
            {
                TreeNodeIndex globalIndex    = globalTree.locate(localLeaves[i], localLeaves[i + 1]);
                TreeNodeIndex internalIdx    = tree_.toInternal(i);
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
        centers_.resize(tree_.numTreeNodes());

#pragma omp parallel for schedule(static)
        for (TreeNodeIndex leafIdx = 0; leafIdx < tree_.numLeafNodes(); ++leafIdx)
        {
            //! prepare local leaf centers
            TreeNodeIndex nodeIdx = tree_.toInternal(leafIdx);
            centers_[nodeIdx] =
                massCenter<RealType>(x.data(), y.data(), z.data(), m.data(), layout[leafIdx], layout[leafIdx + 1]);
        }

        //! upsweep with local data in place
        upsweep(tree_.levelRange(), tree_.childOffsets(), centers_.data(), CombineSourceCenter<T>{});
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
        upsweep(tree_.levelRange(), tree_.childOffsets(), centers_.data(), CombineSourceCenter<T>{});
        //! calculate mac radius for each cell based on location of expansion centers
        setMac<T>(tree_.nodeKeys(), centers_, 1.0 / theta_, box);
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
                      gsl::span<const KeyType> globalTreeLeaves,
                      float invThetaEff)
    {
        centers_.resize(tree_.numTreeNodes());
        auto nodeKeys = tree_.nodeKeys();

#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < nodeKeys.size(); ++i)
        {
            //! set centers to geometric centers for min dist Mac
            centers_[i] = computeMinMacR2(nodeKeys[i], invThetaEff, box);
        }

        updateMacs(box, assignment, globalTreeLeaves);
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
    updateMacs(const Box<T>& box, const SpaceCurveAssignment& assignment, gsl::span<const KeyType> globalTreeLeaves)
    {
        KeyType focusStart = globalTreeLeaves[assignment.firstNodeIdx(myRank_)];
        KeyType focusEnd   = globalTreeLeaves[assignment.lastNodeIdx(myRank_)];

        macs_.resize(tree_.numTreeNodes());
        markMacs(tree_, centers_.data(), box, focusStart, focusEnd, macs_.data());

        rebalanceStatus_ |= macCriterion;
    }

    //! @brief update until converged with a simple min-distance MAC
    template<class T, class DeviceVector = std::vector<KeyType>>
    void converge(const Box<T>& box,
                  gsl::span<const KeyType> particleKeys,
                  gsl::span<const int> peers,
                  const SpaceCurveAssignment& assignment,
                  gsl::span<const KeyType> globalTreeLeaves,
                  gsl::span<const unsigned> globalCounts,
                  float invThetaEff,
                  DeviceVector&& scratch = std::vector<KeyType>{})
    {
        int converged = 0;
        while (converged != numRanks_)
        {
            converged = updateTree(peers, assignment, globalTreeLeaves);
            updateCounts(particleKeys, globalTreeLeaves, globalCounts, scratch);
            updateMinMac(box, assignment, globalTreeLeaves, invThetaEff);
            MPI_Allreduce(MPI_IN_PLACE, &converged, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        }
    }

    //! @brief the fully linked traversable octree
    const Octree<KeyType>& octree() const { return tree_; }
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
            size_t iIdx = tree_.toInternal(i);
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
    //! @brief bucket size (ncrit) inside the focus are
    unsigned bucketSize_;

    //! @brief list of peer ranks from last call to updateTree()
    std::vector<int> peers_;
    //! @brief the tree structures that the peers have for the domain of the executing rank (myRank_)
    std::vector<std::vector<KeyType>> treelets_;

    Octree<KeyType> tree_;

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
 * @tparam        Q                an arithmetic type, or compile-time fix-sized arrays thereof
 * @tparam        T                float or double
 * @tparam        F                function object for octree upsweep
 * @param[in]     globalOctree     a global (replicated on all ranks) tree
 * @param[in]     focusTree        octree focused on the executing rank
 * @param[inout]  quantities       an array of length focusTree.octree().numTreeNodes() with cell properties of the
 *                                 locally focused octree
 * @param[in]     upsweepFunction  callable object that will be used to compute internal cell properties of the
 *                                 global tree based on global leaf quantities
 * @param[in]     upsweepArgs      additional arguments that might be required for a tree upsweep, such as expansion
 *                                 centers if Q is a multipole type.
 *
 * This function obtains missing information for tree cell quantities belonging to far-away ranks which are not
 * peer ranks of the executing rank.
 *
 * The data flow is:
 * cell quantities owned by executing rank -> globalLeafQuantities -> global collective communication -> upsweep
 *   -> back-contribution from globalQuantities into @p quantities
 *
 * Precondition:  quantities contains valid data for each cell, including internal cells,
 *                that fall into the focus range of the executing rank
 * Postcondition: each element of quantities corresponding to non-local cells not owned by any of the peer
 *                ranks contains data obtained through global collective communication between ranks
 */
template<class Q, class KeyType, class T, class F, class... UArgs>
void globalFocusExchange(const Octree<KeyType>& globalOctree,
                         const FocusedOctree<KeyType, T>& focusTree,
                         gsl::span<Q> quantities,
                         F&& upsweepFunction,
                         UArgs&&... upsweepArgs)
{
    TreeNodeIndex numGlobalLeaves = globalOctree.numLeafNodes();
    std::vector<Q> globalLeafQuantities(numGlobalLeaves);
    focusTree.template populateGlobal<Q>(globalOctree.treeLeaves(), quantities, globalLeafQuantities);

    //! exchange global leaves
    mpiAllreduce(MPI_IN_PLACE, globalLeafQuantities.data(), numGlobalLeaves, MPI_SUM);

    std::vector<Q> globalQuantities(globalOctree.numTreeNodes());
    scatter(globalOctree.internalOrder(), globalLeafQuantities.data(), globalQuantities.data());
    //! upsweep with the global tree
    upsweepFunction(globalOctree.levelRange(), globalOctree.childOffsets(), globalQuantities.data(), upsweepArgs...);

    //! from the global tree, extract the part that the executing rank was missing
    focusTree.template extractGlobal<Q>(globalOctree, globalQuantities, quantities);
}

} // namespace cstone
