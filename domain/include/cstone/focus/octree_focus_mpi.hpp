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
#include "cstone/primitives/primitives_gpu.h"
#include "cstone/tree/accel_switch.hpp"

namespace cstone
{

//! @brief A fully traversable octree with a local focus
template<class KeyType, class RealType, class Accelerator = CpuTag>
class FocusedOctree
{
    //! @brief A vector template that resides on the hardware specified as Accelerator
    template<class ValueType>
    using AccVector =
        typename AccelSwitchType<Accelerator, std::vector, thrust::device_vector>::template type<ValueType>;

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
        if constexpr (HaveGpu<Accelerator>{})
        {
            std::vector<KeyType> init{0, nodeRange<KeyType>(0)};
            reallocate(leavesAcc_, init.size(), 1.0);
            memcpyH2D(init.data(), init.size(), rawPtr(leavesAcc_));
            octreeAcc_.resize(nNodes(leavesAcc_));
            buildOctreeGpu(rawPtr(leavesAcc_), octreeAcc_.data());

            reallocate(countsAcc_, counts_.size(), 1.0);
            memcpyH2D(counts_.data(), counts_.size(), rawPtr(countsAcc_));
            reallocate(macsAcc_, macs_.size(), 1.0);
            memcpyH2D(macs_.data(), macs_.size(), rawPtr(macsAcc_));
        }

        leaves_ = std::vector<KeyType>{0, nodeRange<KeyType>(0)};
        treeData_.resize(nNodes(leaves_));
        updateInternalTree<KeyType>(leaves_, treeData_.data());
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

        focusTransfer<KeyType>(leaves_, leafCounts_, bucketSize_, myRank_, prevFocusStart, prevFocusEnd, focusStart,
                               focusEnd, enforcedKeys);
        for (int peer : peers_)
        {
            enforcedKeys.push_back(globalTreeLeaves[assignment.firstNodeIdx(peer)]);
            enforcedKeys.push_back(globalTreeLeaves[assignment.lastNodeIdx(peer)]);
        }
        auto uniqueEnd = std::unique(enforcedKeys.begin(), enforcedKeys.end());
        enforcedKeys.erase(uniqueEnd, enforcedKeys.end());

        bool converged;
        if constexpr (HaveGpu<Accelerator>{})
        {
            converged = CombinedUpdate<KeyType>::updateFocusGpu(
                octreeAcc_, leavesAcc_, bucketSize_, focusStart, focusEnd, enforcedKeys,
                {rawPtr(countsAcc_), countsAcc_.size()}, {rawPtr(macsAcc_), macsAcc_.size()});
        }
        else
        {
            converged = CombinedUpdate<KeyType>::updateFocus(treeData_, leaves_, bucketSize_, focusStart, focusEnd,
                                                             enforcedKeys, counts_, macs_);
        }
        downloadOctree();

        translateAssignment<KeyType>(assignment, globalTreeLeaves, leaves_, peers_, myRank_, assignment_);

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
                      DeviceVector&& /*scratch*/ = std::vector<KeyType>{})
    {
        gsl::span<const KeyType> leaves(leaves_);
        std::vector<MPI_Request> treeletRequests;
        exchangeTreelets(peers_, assignment_, leaves, treelets_, treeletRequests);

        leafCounts_.resize(nNodes(leaves_));
        if constexpr (HaveGpu<Accelerator>{})
        {
            reallocateDestructive(leafCountsAcc_, nNodes(leavesAcc_), 1.01);
            TreeNodeIndex numLeafNodes = treeData_.numLeafNodes;

            computeNodeCountsGpu(rawPtr(leavesAcc_), rawPtr(leafCountsAcc_), numLeafNodes, particleKeys.begin(),
                                 particleKeys.end(), std::numeric_limits<unsigned>::max(), false);
            memcpyD2H(rawPtr(leafCountsAcc_), numLeafNodes, leafCounts_.data());
        }
        else
        {
            // local node counts
            assert(std::is_sorted(particleKeys.begin(), particleKeys.end()));
            computeNodeCounts(leaves_.data(), leafCounts_.data(), nNodes(leaves_), particleKeys.data(),
                              particleKeys.data() + particleKeys.size(), std::numeric_limits<unsigned>::max(), true);
        }

        // 1st upsweep with local data
        counts_.resize(treeData_.numNodes);
        scatter<TreeNodeIndex>(leafToInternal(treeData_), leafCounts_.data(), counts_.data());
        upsweep(treeData_.levelRange, treeData_.childOffsets, counts_.data(), NodeCount<unsigned>{});

        // global counts
        auto globalCountIndices = invertRanges(0, assignment_, nNodes(leaves_));

        // particle counts for leaf nodes in treeLeaves() / leafCounts():
        //   Node indices [firstFocusNode:lastFocusNode] got assigned counts from local particles.
        //   Node index ranges listed in requestIndices got assigned counts from peer ranks.
        //   All remaining indices need to get their counts from the global tree.
        //   They are stored in globalCountIndices.
        for (auto ip : globalCountIndices)
        {
            countRequestParticles<KeyType>(globalTreeLeaves, globalCounts, leaves.subspan(ip.start(), ip.count() + 1),
                                           treeData_.prefixes, treeData_.levelRange, gsl::span<unsigned>(counts_));
        }

        // counts from neighboring peers
        MPI_Waitall(int(peers_.size()), treeletRequests.data(), MPI_STATUS_IGNORE);
        constexpr int countTag = static_cast<int>(P2pTags::focusPeerCounts) + 1;
        exchangeTreeletGeneral(peers_, treelets_, assignment_, gsl::span<const KeyType>(treeData_.prefixes),
                               treeData_.levelRange, leafToInternal(treeData_), gsl::span<unsigned>(counts_), countTag);

        // 2nd upsweep with peer and global data present
        upsweep(treeData_.levelRange, treeData_.childOffsets, counts_.data(), NodeCount<unsigned>{});
        gather(leafToInternal(treeData_), counts_.data(), leafCounts_.data());

        if constexpr (HaveGpu<Accelerator>{})
        {
            memcpyH2D(leafCounts_.data(), assignment_[myRank_].start(), rawPtr(leafCountsAcc_));
            memcpyH2D(leafCounts_.data() + assignment_[myRank_].end(), leafCounts_.size() - assignment_[myRank_].end(),
                      rawPtr(leafCountsAcc_) + assignment_[myRank_].end());
            reallocateDestructive(countsAcc_, counts_.size(), 1.01);
            memcpyH2D(counts_.data(), counts_.size(), rawPtr(countsAcc_));
        }

        rebalanceStatus_ |= countsCriterion;
    }

    template<class T>
    void peerExchange(gsl::span<T> quantities, int commTag) const
    {
        exchangeTreeletGeneral<T>(peers_, treelets_, assignment_, gsl::span<const KeyType>(treeData_.prefixes),
                                  treeData_.levelRange, leafToInternal(treeData_), quantities, commTag);
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
        assert(localQuantities.size() == treeData_.numNodes);

        TreeNodeIndex firstGlobalIdx = findNodeAbove(globalLeaves.data(), globalLeaves.size(), prevFocusStart);
        TreeNodeIndex lastGlobalIdx  = findNodeAbove(globalLeaves.data(), globalLeaves.size(), prevFocusEnd);
        // make sure that the focus is resolved exactly in the global tree
        assert(globalLeaves[firstGlobalIdx] == prevFocusStart);
        assert(globalLeaves[lastGlobalIdx] == prevFocusEnd);

        const KeyType* nodeKeys         = treeData_.prefixes.data();
        const TreeNodeIndex* levelRange = treeData_.levelRange.data();

#pragma omp parallel for schedule(static)
        for (TreeNodeIndex globalIdx = firstGlobalIdx; globalIdx < lastGlobalIdx; ++globalIdx)
        {
            TreeNodeIndex localIdx =
                locateNode(globalLeaves[globalIdx], globalLeaves[globalIdx + 1], nodeKeys, levelRange);
            if (localIdx == treeData_.numNodes)
            {
                // If the global tree is fully converged, but the locally focused tree is just being built up
                // for the first time, it's possible that the global tree has a higher resolution than
                // the focused tree.
                continue;
            }
            assert(decodePlaceholderBit(nodeKeys[localIdx]) == globalLeaves[globalIdx]);
            assert(decodePrefixLength(nodeKeys[localIdx]) ==
                   3 * treeLevel(globalLeaves[globalIdx + 1] - globalLeaves[globalIdx]));
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
    void extractGlobal(const KeyType* globalNodeKeys,
                       const TreeNodeIndex* globalLevelRange,
                       gsl::span<const T> globalQuantities,
                       gsl::span<T> localQuantities) const
    {
        const KeyType* localLeaves      = leaves_.data();
        const TreeNodeIndex* toInternal = leafToInternal(treeData_).data();
        //! requestIndices: range of leaf cell indices in the locally focused tree that need global information
        auto requestIndices = invertRanges(0, assignment_, treeData_.numLeafNodes);
        for (auto range : requestIndices)
        {
            //! from global tree, pull in missing elements into locally focused tree
#pragma omp parallel for schedule(static)
            for (TreeNodeIndex i = range.start(); i < range.end(); ++i)
            {
                TreeNodeIndex globalIndex =
                    locateNode(localLeaves[i], localLeaves[i + 1], globalNodeKeys, globalLevelRange);
                TreeNodeIndex internalIdx    = toInternal[i];
                localQuantities[internalIdx] = globalQuantities[globalIndex];
            }
        }
    }

    template<class Tm, class DeviceVector = std::vector<LocalIndex>>
    void updateCenters(const RealType* x,
                       const RealType* y,
                       const RealType* z,
                       const Tm* m,
                       const SpaceCurveAssignment& assignment,
                       const Octree<KeyType>& globalTree,
                       const Box<RealType>& box,
                       DeviceVector&& scratch1 = std::vector<LocalIndex>{},
                       DeviceVector&& scratch2 = std::vector<LocalIndex>{})
    {
        TreeNodeIndex firstIdx           = assignment_[myRank_].start();
        TreeNodeIndex lastIdx            = assignment_[myRank_].end();
        OctreeView<const KeyType> octree = octreeViewAcc();
        TreeNodeIndex numNodes           = octree.numInternalNodes + octree.numLeafNodes;

        globalCenters_.resize(globalTree.numTreeNodes());
        centers_.resize(numNodes);
        reallocateDestructive(centersAcc_, centers_.size(), 1.01);

        if constexpr (HaveGpu<Accelerator>{})
        {
            static_assert(IsDeviceVector<std::decay_t<DeviceVector>>{});
            size_t bytesLayout = (octree.numLeafNodes + 1) * sizeof(LocalIndex);
            size_t osz1        = reallocateBytes(scratch1, bytesLayout);
            auto* d_layout     = reinterpret_cast<LocalIndex*>(rawPtr(scratch1));

            fillGpu(d_layout, d_layout + octree.numLeafNodes + 1, LocalIndex(0));
            exclusiveScanGpu(rawPtr(leafCountsAcc_) + firstIdx, rawPtr(leafCountsAcc_) + lastIdx + 1,
                             d_layout + firstIdx);
            computeLeafSourceCenterGpu(x, y, z, m, octree.leafToInternal + octree.numInternalNodes, octree.numLeafNodes,
                                       d_layout, rawPtr(centersAcc_));
            //! upsweep with local data in place
            upsweepCentersGpu(maxTreeLevel<KeyType>{}, treeData_.levelRange.data(), octree.childOffsets,
                              rawPtr(centersAcc_));
            memcpyD2H(rawPtr(centersAcc_), numNodes, centers_.data());

            reallocateDevice(scratch1, osz1, 1.0);
        }
        else
        {
            //! compute temporary pre-halo exchange particle layout for local particles only
            std::vector<LocalIndex> layout(leafCounts_.size() + 1, 0);
            std::exclusive_scan(leafCounts_.begin() + firstIdx, leafCounts_.begin() + lastIdx + 1,
                                layout.begin() + firstIdx, 0);
#pragma omp parallel for schedule(static)
            for (TreeNodeIndex leafIdx = 0; leafIdx < treeData_.numLeafNodes; ++leafIdx)
            {
                //! prepare local leaf centers
                TreeNodeIndex nodeIdx = octree.leafToInternal[octree.numInternalNodes + leafIdx];
                centers_[nodeIdx]     = massCenter<RealType>(x, y, z, m, layout[leafIdx], layout[leafIdx + 1]);
            }
            //! upsweep with local data in place
            upsweep(treeData_.levelRange, treeData_.childOffsets, centers_.data(), CombineSourceCenter<RealType>{});
        }

        //! exchange information with peer close to focus
        peerExchange<SourceCenterType<RealType>>(centers_, static_cast<int>(P2pTags::focusPeerCenters));
        //! global exchange for the top nodes that are bigger than local domains
        std::vector<SourceCenterType<RealType>> globalLeafCenters(globalTree.numLeafNodes());
        populateGlobal<SourceCenterType<RealType>>(globalTree.treeLeaves(), centers_, globalLeafCenters);
        mpiAllreduce(MPI_IN_PLACE, globalLeafCenters.data(), globalLeafCenters.size(), MPI_SUM);
        scatter(globalTree.internalOrder(), globalLeafCenters.data(), globalCenters_.data());
        upsweep(globalTree.levelRange(), globalTree.childOffsets(), globalCenters_.data(),
                CombineSourceCenter<RealType>{});
        extractGlobal<SourceCenterType<RealType>>(globalTree.nodeKeys().data(), globalTree.levelRange().data(),
                                                  globalCenters_, centers_);

        //! upsweep with all (leaf) data in place
        upsweep(treeData_.levelRange, treeData_.childOffsets, centers_.data(), CombineSourceCenter<RealType>{});
        //! calculate mac radius for each cell based on location of expansion centers
        setMac<RealType, KeyType>(treeData_.prefixes, centers_, 1.0 / theta_, box);

        if constexpr (HaveGpu<Accelerator>{}) { memcpyH2D(centers_.data(), centers_.size(), rawPtr(centersAcc_)); }
        // else { omp_copy(centers_.begin(), centers_.end(), centersAcc_.begin()); }
    }

    /*! @brief Update the MAC criteria based on a min distance MAC
     *
     * @tparam    T                float or double
     * @param[in] box              global coordinate bounding box
     * @param[in] assignment       assignment of the global leaf tree to ranks
     * @param[in] globalTreeLeaves global cornerstone leaf tree
     */
    void updateMinMac(const Box<RealType>& box,
                      const SpaceCurveAssignment& assignment,
                      gsl::span<const KeyType> globalTreeLeaves,
                      float invThetaEff)
    {
        centers_.resize(treeData_.numNodes);
        const KeyType* nodeKeys = treeData_.prefixes.data();

#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < treeData_.numNodes; ++i)
        {
            //! set centers to geometric centers for min dist Mac
            centers_[i] = computeMinMacR2(nodeKeys[i], invThetaEff, box);
        }

        updateMacs(box, assignment, globalTreeLeaves);
    }

    /*! @brief Update the MAC criteria based on the vector MAC
     *
     * @param[in] box              global coordinate bounding box
     * @param[in] assignment       assignment of the global leaf tree to ranks
     * @param[in] globalTreeLeaves global cornerstone leaf tree
     */
    void updateMacs(const Box<RealType>& box,
                    const SpaceCurveAssignment& assignment,
                    gsl::span<const KeyType> globalTreeLeaves)
    {
        KeyType focusStart = globalTreeLeaves[assignment.firstNodeIdx(myRank_)];
        KeyType focusEnd   = globalTreeLeaves[assignment.lastNodeIdx(myRank_)];

        macs_.resize(treeData_.numNodes);
        markMacs(treeData_.data(), centers_.data(), box, focusStart, focusEnd, macs_.data());

        if constexpr (HaveGpu<Accelerator>{})
        {
            reallocate(macsAcc_, macs_.size(), 1.01);
            memcpyH2D(macs_.data(), macs_.size(), rawPtr(macsAcc_));
        }

        rebalanceStatus_ |= macCriterion;
    }

    void updateGeoCenters(const Box<RealType>& box)
    {
        reallocate(geoCentersAcc_, treeData_.numNodes, 1.01);
        reallocate(geoSizesAcc_, treeData_.numNodes, 1.01);

        if constexpr (HaveGpu<Accelerator>{})
        {
            computeGeoCentersGpu(rawPtr(octreeAcc_.prefixes), treeData_.numNodes, rawPtr(geoCentersAcc_),
                                 rawPtr(geoSizesAcc_), box);
        }
        else { nodeFpCenters<KeyType>(treeData_.prefixes, geoCentersAcc_.data(), geoSizesAcc_.data(), box); }
    }

    //! @brief update until converged with a simple min-distance MAC
    template<class DeviceVector = std::vector<KeyType>>
    void converge(const Box<RealType>& box,
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

    //! @brief returns the tree depth
    TreeNodeIndex depth() const { return maxDepth(treeData_.levelRange.data(), treeData_.levelRange.size()); }

    //! @brief the cornerstone leaf cell array
    gsl::span<const KeyType> treeLeaves() const { return leaves_; }
    //! @brief the assignment of the focus tree leaves to peer ranks
    gsl::span<const TreeIndexPair> assignment() const { return assignment_; }
    //! @brief Expansion (com) centers of each cell
    gsl::span<const SourceCenterType<RealType>> expansionCenters() const { return centers_; }
    gsl::span<const SourceCenterType<RealType>> expansionCentersAcc() const
    {
        return {rawPtr(centersAcc_), centersAcc_.size()};
    }
    //! @brief Expansion (com) centers of each global cell
    gsl::span<const SourceCenterType<RealType>> globalExpansionCenters() const { return globalCenters_; }
    //! brief particle counts per focus tree leaf cell
    gsl::span<const unsigned> leafCounts() const { return leafCounts_; }

    //! @brief return a view to the octree on the active accelerator
    OctreeView<const KeyType> octreeViewAcc() const
    {
        if constexpr (HaveGpu<Accelerator>{}) { return ((const decltype(octreeAcc_)&)octreeAcc_).data(); }
        else { return treeData_.data(); }
    }

    //! @brief the cornerstone leaf cell array on the accelerator
    gsl::span<const KeyType> treeLeavesAcc() const
    {
        if constexpr (HaveGpu<Accelerator>{}) { return {rawPtr(leavesAcc_), leavesAcc_.size()}; }
        else { return leaves_; }
    }

    //! @brief the cornerstone leaf cell particle counts
    gsl::span<const unsigned> leafCountsAcc() const
    {
        if constexpr (HaveGpu<Accelerator>{}) { return {rawPtr(leafCountsAcc_), leafCountsAcc_.size()}; }
        else { return leafCounts_; }
    }

    gsl::span<const Vec3<RealType>> geoCentersAcc() const { return {rawPtr(geoCentersAcc_), geoCentersAcc_.size()}; }
    gsl::span<const Vec3<RealType>> geoSizesAcc() const { return {rawPtr(geoSizesAcc_), geoSizesAcc_.size()}; }

    void addMacs(gsl::span<int> haloFlags) const
    {
        const TreeNodeIndex* toInternal = leafToInternal(treeData_).data();
#pragma omp parallel for schedule(static)
        for (TreeNodeIndex i = 0; i < haloFlags.ssize(); ++i)
        {
            size_t iIdx = toInternal[i];
            if (macs_[iIdx] && !haloFlags[i]) { haloFlags[i] = 1; }
        }
    }

private:
    void uploadOctree()
    {
        if constexpr (HaveGpu<Accelerator>{})
        {
            TreeNodeIndex numLeafNodes = treeData_.numLeafNodes;
            TreeNodeIndex numNodes     = treeData_.numNodes;

            octreeAcc_.resize(numLeafNodes);
            reallocateDestructive(leavesAcc_, numLeafNodes + 1, 1.01);

            memcpyH2D(treeData_.prefixes.data(), numNodes, rawPtr(octreeAcc_.prefixes));
            memcpyH2D(treeData_.childOffsets.data(), numNodes, rawPtr(octreeAcc_.childOffsets));
            memcpyH2D(treeData_.parents.data(), treeData_.parents.size(), rawPtr(octreeAcc_.parents));
            memcpyH2D(treeData_.levelRange.data(), treeData_.levelRange.size(), rawPtr(octreeAcc_.levelRange));
            memcpyH2D(treeData_.internalToLeaf.data(), numNodes, rawPtr(octreeAcc_.internalToLeaf));
            memcpyH2D(treeData_.leafToInternal.data(), numNodes, rawPtr(octreeAcc_.leafToInternal));

            memcpyH2D(leaves_.data(), numLeafNodes + 1, rawPtr(leavesAcc_));
        }
    }

    void downloadOctree()
    {
        if constexpr (HaveGpu<Accelerator>{})
        {
            TreeNodeIndex numLeafNodes = octreeAcc_.numLeafNodes;
            TreeNodeIndex numNodes     = octreeAcc_.numNodes;

            treeData_.resize(numLeafNodes);
            reallocateDestructive(leaves_, numLeafNodes + 1, 1.01);

            memcpyD2H(rawPtr(octreeAcc_.prefixes), numNodes, treeData_.prefixes.data());
            memcpyD2H(rawPtr(octreeAcc_.childOffsets), numNodes, treeData_.childOffsets.data());
            memcpyD2H(rawPtr(octreeAcc_.parents), octreeAcc_.parents.size(), treeData_.parents.data());
            memcpyD2H(rawPtr(octreeAcc_.levelRange), octreeAcc_.levelRange.size(), treeData_.levelRange.data());
            memcpyD2H(rawPtr(octreeAcc_.internalToLeaf), numNodes, treeData_.internalToLeaf.data());
            memcpyD2H(rawPtr(octreeAcc_.leafToInternal), numNodes, treeData_.leafToInternal.data());

            memcpyD2H(rawPtr(leavesAcc_), numLeafNodes + 1, leaves_.data());
        }
    }

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

    //! @brief octree data resident on GPU if active
    OctreeData<KeyType, Accelerator> octreeAcc_;
    AccVector<KeyType> leavesAcc_;
    AccVector<unsigned> leafCountsAcc_;
    AccVector<SourceCenterType<RealType>> centersAcc_;
    AccVector<Vec3<RealType>> geoCentersAcc_;
    AccVector<Vec3<RealType>> geoSizesAcc_;
    AccVector<unsigned> countsAcc_;
    AccVector<char> macsAcc_;

    OctreeData<KeyType, CpuTag> treeData_;
    //! @brief leaves in cstone format for tree_
    std::vector<KeyType> leaves_;

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
template<class Q, class KeyType, class T, class F, class Accelerator, class... UArgs>
void globalFocusExchange(const Octree<KeyType>& globalOctree,
                         const FocusedOctree<KeyType, T, Accelerator>& focusTree,
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
    focusTree.template extractGlobal<Q>(globalOctree.nodeKeys().data(), globalOctree.levelRange().data(),
                                        globalQuantities, quantities);
}

} // namespace cstone
