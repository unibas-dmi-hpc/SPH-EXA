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
 * @brief Request counts for a locally present node structure of a remote domain from a remote rank
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * Overall procedure for each pair for peer ranks (rank1, rank2):
 *      1.  rank1 sends a node structure (vector of SFC keys) to rank2. The node structure sent by rank1
 *          covers the assigned domain of rank2. The node structure cannot exceed the resolution of
 *          the local tree of rank2, this is guaranteed by the tree-build process, as long as all ranks
 *          use the same bucket size for the locally focused tree. Usually, rank1 requests the full resolution
 *          along the surface with rank2 and a lower resolution far a way from the surface.
 *
 *      2.  rank2 receives the the node structure, counts particles for each received node and sends back
 *          an answer with the particle counts per node.
 *
 *      3. rank1 receives the counts for the requested SFC keys from rank2
 */

#pragma once

#include <algorithm>
#include <vector>

#include "cstone/domain/index_ranges.hpp"
#include "cstone/primitives/concat_vector.hpp"
#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/primitives/mpi_cuda.cuh"
#include "cstone/primitives/gather_acc.hpp"
#include "cstone/tree/csarray.hpp"
#include "cstone/tree/csarray_gpu.h"
#include "cstone/tree/octree.hpp"
#include "cstone/util/gsl-lite.hpp"
#include "cstone/util/pack_buffers.hpp"

namespace cstone
{

/*! @brief exchange subtree structures with peers
 *
 * @tparam      KeyType          32- or 64-bit unsigned integer
 * @param[in]   peerRanks        List of peer rank IDs
 * @param[in]   focusAssignment  The assignment of @p localLeaves to peer ranks
 * @param[in]   leaves      The tree of the executing rank. Covers the global domain, but is locally focused.
 * @param[out]  treelets        The tree structures of REMOTE peer ranks covering the LOCALLY assigned part of
 *                               the tree. Each treelet covers the same SFC key range (the assigned range of
 *                               the executing rank) but is adaptively (MAC) resolved from the perspective of the
 *                               peer rank.
 *
 * Note: peerTrees stores the view of REMOTE ranks for the LOCAL domain. While focusAssignment and localLeaves
 * contain the LOCAL view of REMOTE peer domains.
 */
template<class KeyType>
void exchangeTreelets(gsl::span<const int> peerRanks,
                      gsl::span<const IndexPair<TreeNodeIndex>> focusAssignment,
                      gsl::span<const KeyType> leaves,
                      std::vector<std::vector<KeyType>>& treelets)
{
    constexpr int keyTag = static_cast<int>(P2pTags::focusTreelets);
    size_t numPeers      = peerRanks.size();

    std::vector<MPI_Request> sendRequests;
    sendRequests.reserve(numPeers);
    for (auto peer : peerRanks)
    {
        // +1 to include the upper key boundary for the last node
        TreeNodeIndex sendCount = focusAssignment[peer].count() + 1;
        mpiSendAsync(leaves.data() + focusAssignment[peer].start(), sendCount, peer, keyTag, sendRequests);
    }

    std::vector<MPI_Request> receiveRequests;
    receiveRequests.reserve(numPeers);
    int numMessages = numPeers;
    while (numMessages--)
    {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, keyTag, MPI_COMM_WORLD, &status);
        int receiveRank = status.MPI_SOURCE;
        TreeNodeIndex receiveSize;
        MPI_Get_count(&status, MpiType<KeyType>{}, &receiveSize);
        treelets[receiveRank].resize(receiveSize);

        mpiRecvAsync(treelets[receiveRank].data(), receiveSize, receiveRank, keyTag, receiveRequests);
    }

    MPI_Waitall(int(numPeers), sendRequests.data(), MPI_STATUS_IGNORE);
    MPI_Waitall(int(numPeers), receiveRequests.data(), MPI_STATUS_IGNORE);
}

//! @brief flag treelet keys that don't exist in @p leaves as invalid
template<class KeyType>
void checkTreelets(gsl::span<const int> peerRanks,
                   gsl::span<const KeyType> leaves,
                   std::vector<std::vector<KeyType>>& treelets)
{
    for (auto rank : peerRanks)
    {
        auto& treelet          = treelets[rank];
        TreeNodeIndex numNodes = nNodes(treelets[rank]);

#pragma omp parallel for schedule(static)
        for (int i = 0; i < numNodes; ++i)
        {
            auto k = treelet[i];
            if (k != leaves[findNodeAbove(leaves.data(), nNodes(leaves), k)]) { treelet[i] = maskKey(k); }
        }
    }
}

//! @brief remove treelet keys flagged as invalid
template<class KeyType>
void pruneTreelets(gsl::span<const int> peerRanks, std::vector<std::vector<KeyType>>& treelets)
{
#pragma omp parallel for
    for (int r = 0; r < peerRanks.size(); ++r)
    {
        int rank = peerRanks[r];
        auto it  = std::remove_if(treelets[rank].begin(), treelets[rank].end(), isMasked<KeyType>);
        treelets[rank].erase(it, treelets[rank].end());
    }
}

/*! @brief exchange subtree structures with peers
 *
 * @tparam      KeyType      32- or 64-bit unsigned integer
 * @param[in]   peerRanks    List of peer rank IDs
 * @param[in]   leaves       leaves of the LET
 * @param[in]   treelets     The tree structures of REMOTE peer ranks covering the LOCALLY assigned part of
 *                           the tree. Each treelet covers the same SFC key range (the assigned range of
 *                           the executing rank) but is adaptively (MAC) resolved from the perspective of the
 *                           peer rank.
 * @param[out]  nodeOps      node ops needed to remove exterior keys that don't exist on the owning rank from
 *                           @p leaves
 *
 * Note: peerTrees stores the view of REMOTE ranks for the LOCAL domain. While focusAssignment and localLeaves
 * contain the LOCAL view of REMOTE peer domains.
 */
template<class KeyType>
void exchangeRejectedKeys(gsl::span<const int> peerRanks,
                          gsl::span<const KeyType> leaves,
                          const std::vector<std::vector<KeyType>>& treelets,
                          gsl::span<TreeNodeIndex> nodeOps)

{
    constexpr int keyTag = static_cast<int>(P2pTags::focusTreelets) + 1;
    size_t numPeers      = peerRanks.size();

    std::vector<MPI_Request> sendRequests;
    sendRequests.reserve(numPeers);

    std::vector<std::vector<KeyType, util::DefaultInitAdaptor<KeyType>>> rejectedKeyBuffers;
    for (auto peer : peerRanks)
    {
        auto& treelet          = treelets[peer];
        TreeNodeIndex numNodes = nNodes(treelet);

        std::vector<KeyType, util::DefaultInitAdaptor<KeyType>> rejectedKeys;
        for (int i = 0; i < numNodes; ++i)
        {
            if (isMasked(treelet[i])) { rejectedKeys.push_back(unmaskKey(treelet[i])); }
        }
        mpiSendAsync(rejectedKeys.data(), rejectedKeys.size(), peer, keyTag, sendRequests);
        rejectedKeyBuffers.push_back(std::move(rejectedKeys));
    }

    int numMessages = numPeers;
    while (numMessages--)
    {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, keyTag, MPI_COMM_WORLD, &status);
        int receiveRank = status.MPI_SOURCE;
        TreeNodeIndex receiveSize;
        MPI_Get_count(&status, MpiType<KeyType>{}, &receiveSize);

        std::vector<KeyType, util::DefaultInitAdaptor<KeyType>> recvKeys(receiveSize);
        recvKeys.resize(receiveSize);
        mpiRecvSync(recvKeys.data(), receiveSize, receiveRank, keyTag, &status);
        for (TreeNodeIndex i = 0; i < receiveSize; ++i)
        {
            TreeNodeIndex ki = findNodeAbove(leaves.data(), leaves.size(), recvKeys[i]);
            nodeOps[ki]      = 0;
        }
    }

    MPI_Waitall(int(numPeers), sendRequests.data(), MPI_STATUS_IGNORE);
}

template<class KeyType>
void syncTreelets(gsl::span<const int> peers,
                  gsl::span<const IndexPair<TreeNodeIndex>> focusAssignment,
                  OctreeData<KeyType, CpuTag>& octree,
                  std::vector<KeyType>& leaves,
                  std::vector<std::vector<KeyType>>& treelets)
{
    exchangeTreelets<KeyType>(peers, focusAssignment, leaves, treelets);
    checkTreelets<KeyType>(peers, leaves, treelets);

    std::vector<TreeNodeIndex> nodeOps(leaves.size(), 1);
    exchangeRejectedKeys<KeyType>(peers, leaves, treelets, nodeOps);
    pruneTreelets<KeyType>(peers, treelets);

    if (std::count(nodeOps.begin(), nodeOps.end(), 1) != nodeOps.size())
    {
        rebalanceTree(leaves, octree.prefixes, nodeOps.data());
        swap(leaves, octree.prefixes);
        octree.resize(nNodes(leaves));
        updateInternalTree<KeyType>(leaves, octree.data());
    }
}

template<class KeyType>
void syncTreeletsGpu(gsl::span<const int> peers,
                     gsl::span<const IndexPair<TreeNodeIndex>> assignment,
                     gsl::span<const KeyType> leaves,
                     OctreeData<KeyType, GpuTag>& octreeAcc,
                     DeviceVector<KeyType>& leavesAcc,
                     std::vector<std::vector<KeyType>>& treelets)
{
    exchangeTreelets<KeyType>(peers, assignment, leaves, treelets);
    checkTreelets<KeyType>(peers, leaves, treelets);

    std::vector<TreeNodeIndex> nodeOps(leaves.size(), 1);
    exchangeRejectedKeys<KeyType>(peers, leaves, treelets, nodeOps);
    pruneTreelets<KeyType>(peers, treelets);

    if (std::count(nodeOps.begin(), nodeOps.end(), 1) != nodeOps.size())
    {
        assert(octreeAcc.childOffsets.size() >= nodeOps.size());
        gsl::span<TreeNodeIndex> nops(rawPtr(octreeAcc.childOffsets), nodeOps.size());
        memcpyH2D(rawPtr(nodeOps), nodeOps.size(), nops.data());

        exclusiveScanGpu(nops.data(), nops.data() + nops.size(), nops.data());
        TreeNodeIndex newNumLeafNodes;
        memcpyD2H(nops.data() + nops.size() - 1, 1, &newNumLeafNodes);

        auto& newLeaves = octreeAcc.prefixes;
        reallocateDestructive(newLeaves, newNumLeafNodes + 1, 1.05);
        rebalanceTreeGpu(rawPtr(leavesAcc), nNodes(leavesAcc), newNumLeafNodes, nops.data(), rawPtr(newLeaves));
        swap(newLeaves, leavesAcc);

        octreeAcc.resize(nNodes(leavesAcc));
        buildOctreeGpu(rawPtr(leavesAcc), octreeAcc.data());
    }
}

template<class VecOfVec>
std::vector<std::size_t> extractNumNodes(const VecOfVec& vov)
{
    std::vector<std::size_t> ret(vov.size());
    for (size_t i = 0; i < vov.size(); ++i)
    {
        ret[i] = vov[i].empty() ? 0 : nNodes(vov[i]);
    }
    return ret;
}

//! @brief assign treelet nodes their final indices w.r.t the final LET
template<class KeyType>
void indexTreelets(gsl::span<const int> peerRanks, gsl::span<const KeyType> nodeKeys,
                   gsl::span<const TreeNodeIndex> levelRange,
                   const std::vector<std::vector<KeyType>>& treelets,
                   ConcatVector<TreeNodeIndex>& treeletIdx)
{
    auto tlView = treeletIdx.reindex(extractNumNodes(treelets));
    for (int rank : peerRanks)
    {
        const auto& treelet    = treelets[rank];
        auto tlIdx             = tlView[rank];
        TreeNodeIndex numNodes = nNodes(treelets[rank]);

#pragma omp parallel for schedule(static)
        for (int i = 0; i < numNodes; ++i)
        {
            tlIdx[i] = locateNode(treelet[i], treelet[i + 1], nodeKeys.data(), levelRange.data());
            assert(tlIdx[i] < nodeKeys.size());
        }
    }
}

template<class T, class DevVec>
void exchangeTreeletGeneral(gsl::span<const int> peerRanks,
                            gsl::span<const gsl::span<const TreeNodeIndex>> treeletIdx,
                            gsl::span<const IndexPair<TreeNodeIndex>> focusAssignment,
                            gsl::span<const TreeNodeIndex> csToInternalMap,
                            gsl::span<T> quantities,
                            int commTag,
                            DevVec& scratch)
{
    constexpr int alignmentBytes = 64;
    constexpr bool useGpu        = IsDeviceVector<DevVec>{};

    std::vector<std::size_t> treeletSizes(2 * peerRanks.size());
    for (int i = 0; i < peerRanks.size(); ++i)
    {
        treeletSizes[i]                    = treeletIdx[peerRanks[i]].size();       // send buffers
        treeletSizes[i + peerRanks.size()] = focusAssignment[peerRanks[i]].count(); // recv buffers
    }

    size_t origSize    = scratch.size();
    auto packedBuffers = util::packAllocBuffer<T>(scratch, treeletSizes, alignmentBytes);
    gsl::span<gsl::span<T>> sendBuffers{packedBuffers.data(), peerRanks.size()};
    gsl::span<gsl::span<T>> recvBuffers{packedBuffers.data() + peerRanks.size(), peerRanks.size()};

    std::vector<std::vector<T, util::DefaultInitAdaptor<T>>> staging; // only used if GPU-direct is not active
    std::vector<MPI_Request> sendRequests;
    sendRequests.reserve(peerRanks.size());
    for (int i = 0; i < peerRanks.size(); ++i)
    {
        gatherAcc<useGpu, TreeNodeIndex>(treeletIdx[peerRanks[i]], quantities.data(), sendBuffers[i].data());
        if constexpr (useGpu)
        {
            syncGpu();
            mpiSendGpuDirect(sendBuffers[i].data(), treeletIdx[peerRanks[i]].size(), peerRanks[i], commTag,
                             sendRequests, staging);
        }
        else
        {
            mpiSendAsync(sendBuffers[i].data(), treeletIdx[peerRanks[i]].size(), peerRanks[i], commTag, sendRequests);
        }
    }

    int numMessages = peerRanks.size();
    while (numMessages--)
    {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, commTag, MPI_COMM_WORLD, &status);
        int recvRank = status.MPI_SOURCE;
        TreeNodeIndex recvCount;
        mpiGetCount<T>(&status, &recvCount);

        int peerIdx = std::find(peerRanks.begin(), peerRanks.end(), recvRank) - peerRanks.begin();
        T* recvBuf  = recvBuffers[peerIdx].data();
        if constexpr (useGpu) { mpiRecvGpuDirect(recvBuf, recvCount, recvRank, commTag, MPI_STATUS_IGNORE); }
        else { mpiRecvSync(recvBuf, recvCount, recvRank, commTag, MPI_STATUS_IGNORE); }

        auto mapToInternal = csToInternalMap.subspan(focusAssignment[recvRank].start(), recvCount);
        scatterAcc<useGpu>(mapToInternal, recvBuf, quantities.data());
    }
    if constexpr (useGpu) { syncGpu(); }

    MPI_Waitall(int(sendRequests.size()), sendRequests.data(), MPI_STATUS_IGNORE);
    reallocate(scratch, origSize, 1.0);
}

/*! @brief Pass on focus tree parts from old owners to new owners
 *
 * @tparam       KeyType        32- or 64-bit unsigned integer
 * @param[in]    cstree         cornerstone leaf cell array (of the locally focused tree)
 * @param[in]    counts         particle counts per cell in @p cstree, length cstree.size() - 1
 * @param[in]    myRank         the executing rank
 * @param[in]    oldFocusStart  SFC assignment boundaries from previous iteration
 * @param[in]    oldFocusEnd
 * @param[in]    newFocusStart  new SFC assignment boundaries
 * @param[in]    newFocusEnd
 * @param[inout] buffer         cell keys of parts of a remote rank's @p cstree for the newly assigned area of @p myRank
 *                              will be appended to this
 *
 * When the assignment boundaries change, we let the previous owning rank pass on its focus tree of the part that it
 * lost to the new owning rank. Thanks to doing so we can guarantee that each rank always has the highest focus
 * tree resolution inside its focus of any rank: if rank a has focus SFC range F, then no other rank can have
 * tree cells in F that don't exist in rank a's focus tree.
 */
template<class KeyType>
void focusTransfer(gsl::span<const KeyType> cstree,
                   gsl::span<const unsigned> counts,
                   unsigned bucketSize,
                   int myRank,
                   KeyType oldFocusStart,
                   KeyType oldFocusEnd,
                   KeyType newFocusStart,
                   KeyType newFocusEnd,
                   std::vector<KeyType>& buffer)
{
    constexpr int ownerTag = static_cast<int>(P2pTags::focusTransfer);

    std::vector<MPI_Request> sendRequests;
    std::vector<std::vector<KeyType>> sendBuffers;

    if (oldFocusStart < newFocusStart)
    {
        // current rank lost range [oldFocusStart : newFocusStart] to rank below
        TreeNodeIndex start = findNodeAbove(cstree.data(), cstree.size(), oldFocusStart);
        TreeNodeIndex end   = findNodeAbove(cstree.data(), cstree.size(), newFocusStart);

        size_t numNodes = end - start;
        auto treelet    = updateTreelet(gsl::span<const KeyType>(cstree.data() + start, numNodes + 1),
                                        gsl::span<const unsigned>(counts.data() + start, numNodes), bucketSize);

        mpiSendAsync(treelet.data(), int(treelet.size() - 1), myRank - 1, ownerTag, sendRequests);
        sendBuffers.push_back(std::move(treelet));
    }

    if (newFocusEnd < oldFocusEnd)
    {
        // current rank lost range [newFocusEnd : oldFocusEnd] to rank above
        TreeNodeIndex start = findNodeAbove(cstree.data(), cstree.size(), newFocusEnd);
        TreeNodeIndex end   = findNodeAbove(cstree.data(), cstree.size(), oldFocusEnd);

        size_t numNodes = end - start;
        auto treelet    = updateTreelet(gsl::span<const KeyType>(cstree.data() + start, numNodes + 1),
                                        gsl::span<const unsigned>(counts.data() + start, numNodes), bucketSize);

        mpiSendAsync(treelet.data(), int(treelet.size() - 1), myRank + 1, ownerTag, sendRequests);
        sendBuffers.push_back(std::move(treelet));
    }

    if (newFocusStart < oldFocusStart)
    {
        // current rank gained range [newFocusStart : oldFocusStart] from rank below
        MPI_Status status;
        MPI_Probe(myRank - 1, ownerTag, MPI_COMM_WORLD, &status);
        TreeNodeIndex receiveSize;
        MPI_Get_count(&status, MpiType<KeyType>{}, &receiveSize);

        buffer.resize(buffer.size() + receiveSize);
        mpiRecvSync(buffer.data() + buffer.size() - receiveSize, receiveSize, myRank - 1, ownerTag, &status);
    }

    if (oldFocusEnd < newFocusEnd)
    {
        // current rank gained range [oldFocusEnd : newFocusEnd] from rank above
        MPI_Status status;
        MPI_Probe(myRank + 1, ownerTag, MPI_COMM_WORLD, &status);
        TreeNodeIndex receiveSize;
        MPI_Get_count(&status, MpiType<KeyType>{}, &receiveSize);

        buffer.resize(buffer.size() + receiveSize);
        mpiRecvSync(buffer.data() + buffer.size() - receiveSize, receiveSize, myRank + 1, ownerTag, MPI_STATUS_IGNORE);
    }

    MPI_Waitall(int(sendRequests.size()), sendRequests.data(), MPI_STATUS_IGNORE);
}

} // namespace cstone
