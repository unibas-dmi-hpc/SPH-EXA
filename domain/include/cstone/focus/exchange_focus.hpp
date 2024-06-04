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

#include <vector>

#include "cstone/domain/index_ranges.hpp"
#include "cstone/primitives/gather.hpp"
#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/tree/csarray.hpp"
#include "cstone/tree/octree.hpp"
#include "cstone/util/gsl-lite.hpp"

namespace cstone
{

/*! @brief count particles inside specified ranges of a cornerstone leaf tree
 *
 * @tparam KeyType             32- or 64-bit unsigned integer
 * @param[in]  leaves          cornerstone SFC key sequence,
 * @param[in]  counts          particle counts of @p leaves, length = length(leaves) - 1
 * @param[in]  requestLeaves   query cornerstone SFC key sequence
 * @param[in]  prefixes        node keys that matches the layout of @p request counts
 * @param[in]  levelRange      octree-level boundary starts, used for efficiently locating nodes @p prefixes
 * @param[out] requestCounts   output counts for @p requestLeaves, length = same as @p prefixes
 */
template<class KeyType>
void countRequestParticles(gsl::span<const KeyType> leaves,
                           gsl::span<const unsigned> counts,
                           gsl::span<const KeyType> requestLeaves,
                           gsl::span<const KeyType> prefixes,
                           gsl::span<const TreeNodeIndex> levelRange,
                           gsl::span<unsigned> requestCounts)
{
#pragma omp parallel for
    for (size_t i = 0; i < nNodes(requestLeaves); ++i)
    {
        KeyType startKey = requestLeaves[i];
        KeyType endKey   = requestLeaves[i + 1];

        size_t startIdx = findNodeBelow(leaves.data(), leaves.size(), startKey);
        size_t endIdx   = findNodeAbove(leaves.data(), leaves.size(), endKey);

        TreeNodeIndex internalIdx = locateNode(startKey, endKey, prefixes.data(), levelRange.data());

        // Nodes in @p leaves must match the request keys exactly, otherwise counts are wrong.
        // If this assertion fails, it means that the local leaves/counts does not have the required
        // resolution to answer the incoming count request of [startKey:endKey] precisely, which means that
        // focusTransfer didn't work correctly
        assert(startKey == leaves[startIdx]);
        assert(endKey == leaves[endIdx]);

        uint64_t internalCount     = std::accumulate(counts.begin() + startIdx, counts.begin() + endIdx, uint64_t(0));
        requestCounts[internalIdx] = std::min(uint64_t(std::numeric_limits<unsigned>::max()), internalCount);
    }
}

/*! @brief exchange subtree structures with peers
 *
 * @tparam      KeyType          32- or 64-bit unsigned integer
 * @param[in]   peerRanks        List of peer rank IDs
 * @param[in]   focusAssignment  The assignment of @p localLeaves to peer ranks
 * @param[in]   leaves           Like @p prefixes, but just the leaves.
 * @param[in]   prefixes         The full tree of the executing rank. Covers the global domain, but is locally focused.
 * @param[in]   levelRange       Node index ranges of each tree-level of the octree described by @p prefixes
 * @param[out]  peerTrees        The tree structures of REMOTE peer ranks covering the LOCALLY assigned part of
 *                               the tree. Each treelet covers the same SFC key range (the assigned range of
 *                               the executing rank) but is adaptively (MAC) resolved from the perspective of the
 *                               peer rank. It is indexed w.r.t. @p localPrefixes
 * @param[out]  nodeOps          node ops needed to remove exterior keys that don't exist on the owning rank from
 *                               @p leaves
 *
 * Note: peerTrees stores the view of REMOTE ranks for the LOCAL domain. While focusAssignment and localLeaves
 * contain the LOCAL view of REMOTE peer domains.
 */
template<class KeyType>
void exchangeTreelets(gsl::span<const int> peerRanks,
                      gsl::span<const IndexPair<TreeNodeIndex>> focusAssignment,
                      gsl::span<const KeyType> leaves,
                      gsl::span<const KeyType> prefixes,
                      gsl::span<const TreeNodeIndex> levelRange,
                      std::vector<std::vector<TreeNodeIndex>>& peerTrees,
                      gsl::span<TreeNodeIndex> nodeOps)

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

    std::vector<std::vector<KeyType, util::DefaultInitAdaptor<KeyType>>> rejectedKeyBuffers;

    std::vector<KeyType, util::DefaultInitAdaptor<KeyType>> treelet;
    int numMessages = numPeers;
    while (numMessages--)
    {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, keyTag, MPI_COMM_WORLD, &status);
        int receiveRank = status.MPI_SOURCE;
        TreeNodeIndex receiveSize;
        MPI_Get_count(&status, MpiType<KeyType>{}, &receiveSize);

        treelet.clear();
        treelet.resize(receiveSize);
        mpiRecvSync(treelet.data(), receiveSize, receiveRank, keyTag, &status);

        TreeNodeIndex numNodes = nNodes(treelet);
        auto& pTree            = peerTrees[receiveRank];
        pTree.resize(numNodes);

#pragma omp parallel for
        for (int i = 0; i < numNodes; ++i)
        {
            TreeNodeIndex internalIdx = locateNode(treelet[i], treelet[i + 1], prefixes.data(), levelRange.data());
            if (internalIdx >= prefixes.size())
            {
                TreeNodeIndex enclLeaf = findNodeAbove(leaves.data(), nNodes(leaves), treelet[i]);
                if (treelet[i] == leaves[enclLeaf])
                {
                    pTree[i] = locateNode(leaves[enclLeaf], leaves[enclLeaf + 1], prefixes.data(), levelRange.data());
                }
                else { pTree[i] = -1; }
            }
            else { pTree[i] = internalIdx; }
        }

        // prepare response to interior peers
        std::vector<KeyType, util::DefaultInitAdaptor<KeyType>> rejectedKeys;
        for (int i = 0; i < numNodes; ++i)
        {
            if (pTree[i] < 0) { rejectedKeys.push_back(treelet[i]); }
        }
        mpiSendAsync(rejectedKeys.data(), rejectedKeys.size(), receiveRank, keyTag + 1, sendRequests);
        rejectedKeyBuffers.push_back(std::move(rejectedKeys));

        // prune local treelets of interior peers
        auto it = std::remove_if(pTree.begin(), pTree.end(), [](auto x) { return x < 0; });
        pTree.erase(it, pTree.end());
    }

    // receive rejected keys to fix exterior trees
    numMessages = numPeers;
    while (numMessages--)
    {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, keyTag + 1, MPI_COMM_WORLD, &status);
        int receiveRank = status.MPI_SOURCE;
        TreeNodeIndex receiveSize;
        MPI_Get_count(&status, MpiType<KeyType>{}, &receiveSize);

        auto& rejectedKeys = treelet;
        rejectedKeys.resize(receiveSize);
        mpiRecvSync(rejectedKeys.data(), receiveSize, receiveRank, keyTag + 1, &status);
        for (TreeNodeIndex i = 0; i < receiveSize; ++i)
        {
            TreeNodeIndex ki = findNodeAbove(leaves.data(), leaves.size(), rejectedKeys[i]);
            nodeOps[ki]      = 0;
        }
    }

    MPI_Waitall(int(numPeers), sendRequests.data(), MPI_STATUS_IGNORE);
}

template<class T>
void exchangeTreeletGeneral(gsl::span<const int> peerRanks,
                            const std::vector<std::vector<TreeNodeIndex>>& peerTrees,
                            gsl::span<const IndexPair<TreeNodeIndex>> focusAssignment,
                            gsl::span<const TreeNodeIndex> csToInternalMap,
                            gsl::span<T> quantities,
                            int commTag)
{
    size_t numPeers = peerRanks.size();
    std::vector<std::vector<T, util::DefaultInitAdaptor<T>>> sendBuffers;
    sendBuffers.reserve(numPeers);

    std::vector<MPI_Request> sendRequests;
    sendRequests.reserve(numPeers);
    for (auto peer : peerRanks)
    {
        std::vector<T, util::DefaultInitAdaptor<T>> buffer(peerTrees[peer].size());
        gsl::span<const TreeNodeIndex> treelet(peerTrees[peer]);
        gather<TreeNodeIndex>(treelet, quantities.data(), buffer.data());

        mpiSendAsync(buffer.data(), buffer.size(), peer, commTag, sendRequests);
        sendBuffers.push_back(std::move(buffer));
    }

    std::vector<T, util::DefaultInitAdaptor<T>> buffer;
    for (auto peer : peerRanks)
    {
        TreeNodeIndex receiveCount = focusAssignment[peer].count();
        buffer.resize(receiveCount);
        mpiRecvSync(buffer.data(), receiveCount, peer, commTag, MPI_STATUS_IGNORE);

        auto mapToInternal = csToInternalMap.subspan(focusAssignment[peer].start(), receiveCount);
        scatter(mapToInternal, buffer.data(), quantities.data());
    }

    MPI_Waitall(int(sendRequests.size()), sendRequests.data(), MPI_STATUS_IGNORE);
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
