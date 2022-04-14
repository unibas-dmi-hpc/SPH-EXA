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

#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/tree/octree.hpp"
#include "cstone/util/gsl-lite.hpp"
#include "cstone/util/index_ranges.hpp"

namespace cstone
{

/*! @brief count particles inside specified ranges of a cornerstone leaf tree
 *
 * @tparam KeyType             32- or 64-bit unsigned integer
 * @param[in]  leaves          cornerstone SFC key sequence,
 * @param[in]  counts          particle counts of @p leaves, length = length(leaves) - 1
 * @param[in]  requestLeaves   query cornerstone SFC key sequence
 * @param[out] requestCounts   output counts for @p requestLeaves, length = length(requestLeaves) - 1
 */
template<class KeyType>
void countRequestParticles(gsl::span<const KeyType> leaves,
                           gsl::span<const unsigned> counts,
                           gsl::span<const KeyType> requestLeaves,
                           gsl::span<unsigned> requestCounts)
{
#pragma omp parallel for
    for (size_t i = 0; i < requestCounts.size(); ++i)
    {
        KeyType startKey = requestLeaves[i];
        KeyType endKey   = requestLeaves[i + 1];

        size_t startIdx = findNodeBelow(leaves, startKey);
        size_t endIdx   = findNodeAbove(leaves, endKey);

        // Nodes in @p leaves must match the request keys exactly, otherwise counts are wrong.
        // If this assertion fails, it means that the local leaves/counts does not have the required
        // resolution to answer the incoming count request of [startKey:endKey] precisely, which means that
        // focusTransfer didn't work correctly
        assert(startKey == leaves[startIdx]);
        assert(endKey == leaves[endIdx]);

        requestCounts[i] = std::accumulate(counts.begin() + startIdx, counts.begin() + endIdx, 0u);
    }
}

/*! @brief exchange subtree structures with peers
 *
 * @tparam      KeyType          32- or 64-bit unsigned integer
 * @param[in]   peerRanks        List of peer rank IDs
 * @param[in]   focusAssignment  The assignment of @p localLeaves to peer ranks
 * @param[in]   localLeaves      The tree of the executing rank. Covers the global domain, but is locally focused.
 * @param[out]  peerTrees        The tree structures of REMOTE peer ranks covering the LOCALLY assigned part of
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
                      gsl::span<const KeyType> localLeaves,
                      std::vector<std::vector<KeyType>>& peerTrees,
                      std::vector<MPI_Request>& receiveRequests)

{
    constexpr int keyTag = static_cast<int>(P2pTags::focusPeerCounts);
    size_t numPeers      = peerRanks.size();

    std::vector<MPI_Request> sendRequests;
    sendRequests.reserve(numPeers);
    for (auto peer : peerRanks)
    {
        // +1 to include the upper key boundary for the last node
        TreeNodeIndex sendCount = focusAssignment[peer].count() + 1;
        mpiSendAsync(localLeaves.data() + focusAssignment[peer].start(), sendCount, peer, keyTag, sendRequests);
    }

    receiveRequests.reserve(numPeers);
    int numMessages = numPeers;
    while (numMessages > 0)
    {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, keyTag, MPI_COMM_WORLD, &status);
        int receiveRank = status.MPI_SOURCE;
        TreeNodeIndex receiveSize;
        MPI_Get_count(&status, MpiType<KeyType>{}, &receiveSize);
        peerTrees[receiveRank].resize(receiveSize);

        mpiRecvAsync(peerTrees[receiveRank].data(), receiveSize, receiveRank, keyTag, receiveRequests);

        numMessages--;
    }

    MPI_Waitall(int(numPeers), sendRequests.data(), MPI_STATUS_IGNORE);
}

template<class KeyType>
void exchangeTreeletCounts(gsl::span<const int> peerRanks,
                           const std::vector<std::vector<KeyType>>& peerTrees,
                           gsl::span<const IndexPair<TreeNodeIndex>> focusAssignment,
                           gsl::span<const KeyType> localLeaves,
                           gsl::span<unsigned> localCounts,
                           std::vector<MPI_Request>& treeletRequests)
{
    size_t numPeers = peerRanks.size();
    std::vector<std::vector<unsigned>> sendBuffers;
    sendBuffers.reserve(numPeers);

    constexpr int countTag = static_cast<int>(P2pTags::focusPeerCounts) + 1;

    std::vector<MPI_Request> sendRequests;
    sendRequests.reserve(numPeers);
    int numMessages = numPeers;
    while (numMessages > 0)
    {
        MPI_Status status;
        int requestIdx;
        MPI_Waitany(numPeers, treeletRequests.data(), &requestIdx, &status);
        int peer = status.MPI_SOURCE;
        // compute particle counts for the remote peer's tree structure of the local domain.
        TreeNodeIndex numNodes = nNodes(peerTrees[peer]);
        std::vector<unsigned> countBuffer(numNodes);
        countRequestParticles<KeyType>(localLeaves, localCounts, peerTrees[peer], countBuffer);

        // send back answer with the counts for the requested nodes
        mpiSendAsync(countBuffer.data(), numNodes, peer, countTag, sendRequests);
        sendBuffers.push_back(std::move(countBuffer));

        numMessages--;
    }

    treeletRequests.clear();
    for (auto peer : peerRanks)
    {
        TreeNodeIndex receiveCount = focusAssignment[peer].count();
        mpiRecvAsync(localCounts.data() + focusAssignment[peer].start(), receiveCount, peer, countTag, treeletRequests);
    }

    MPI_Waitall(int(sendRequests.size()), sendRequests.data(), MPI_STATUS_IGNORE);
}

template<class T, class KeyType>
void exchangeTreeletGeneral(gsl::span<const int> peerRanks,
                            const std::vector<std::vector<KeyType>>& peerTrees,
                            gsl::span<const IndexPair<TreeNodeIndex>> focusAssignment,
                            gsl::span<const KeyType> octree,
                            gsl::span<const TreeNodeIndex> levelRange,
                            gsl::span<const TreeNodeIndex> csToInternalMap,
                            gsl::span<T> quantities,
                            int commTag)
{
    size_t numPeers = peerRanks.size();
    std::vector<std::vector<T>> sendBuffers;
    sendBuffers.reserve(numPeers);

    std::vector<MPI_Request> sendRequests;
    sendRequests.reserve(numPeers);
    for (auto peer : peerRanks)
    {
        TreeNodeIndex treeletSize = nNodes(peerTrees[peer]);
        std::vector<T> buffer(treeletSize);
        gsl::span<const KeyType> treelet(peerTrees[peer]);

        for (TreeNodeIndex i = 0; i < treeletSize; ++i)
        {
            KeyType nodeStart = treelet[i];
            KeyType nodeEnd   = treelet[i + 1];
            unsigned level    = treeLevel(nodeEnd - nodeStart);
            KeyType prefix    = encodePlaceholderBit(nodeStart, 3 * level);

            TreeNodeIndex internalIdx =
                stl::lower_bound(octree.data() + levelRange[level], octree.data() + levelRange[level + 1], prefix) -
                octree.data();
            assert(prefix == octree[internalIdx]);

            buffer[i] = quantities[internalIdx];
        }

        mpiSendAsync(buffer.data(), treeletSize, peer, commTag, sendRequests);
        sendBuffers.push_back(std::move(buffer));
    }

    std::vector<T> buffer;
    for (auto peer : peerRanks)
    {
        TreeNodeIndex receiveCount = focusAssignment[peer].count();
        buffer.resize(receiveCount);
        mpiRecvSync(buffer.data(), receiveCount, peer, commTag, MPI_STATUS_IGNORE);
        for (int i = 0; i < receiveCount; ++i)
        {
            TreeNodeIndex csIdx       = i + focusAssignment[peer].start();
            TreeNodeIndex internalIdx = csToInternalMap[csIdx];
            quantities[internalIdx]   = buffer[i];
        }
    }

    MPI_Waitall(int(sendRequests.size()), sendRequests.data(), MPI_STATUS_IGNORE);
}

/*! @brief exchange particle counts with specified peer ranks
 *
 * @tparam KeyType                  32- or 64-bit unsigned integer
 * @param[in]  peerRanks            list of peer rank IDs
 * @param[in]  exchangeIndices      contains one range of indices of @p localLeaves to request counts for each peer rank
 *                                  length = numRanks
 * @param[in]  localLeaves          cornerstone SFC key sequence of the locally (focused) tree
 *                                  of the executing rank
 * @param[out] localCounts          particle counts associated with @p localLeaves
 *                                  length(localCounts) = length(localLeaves) - 1
 *
 * Procedure on each rank:
 *  1. Send out the SFC keys for which it wants to get particle counts to peer ranks
 *  2. receive SFC keys from other peer ranks, count particles and send back the counts as answer
 *  3. receive answer with the counts for the requested keys
 */
template<class KeyType>
void exchangePeerCounts(gsl::span<const int> peerRanks,
                        gsl::span<const IndexPair<TreeNodeIndex>> exchangeIndices,
                        gsl::span<const KeyType> localLeaves,
                        gsl::span<unsigned> localCounts)

{
    std::vector<std::vector<KeyType>> treelets(exchangeIndices.size());
    std::vector<MPI_Request> treeletRequests;

    exchangeTreelets(peerRanks, exchangeIndices, localLeaves, treelets, treeletRequests);
    exchangeTreeletCounts(peerRanks, treelets, exchangeIndices, localLeaves, localCounts, treeletRequests);

    MPI_Waitall(int(peerRanks.size()), treeletRequests.data(), MPI_STATUS_IGNORE);

    // MUST call MPI_Barrier or any other collective MPI operation that enforces synchronization
    // across all ranks before calling this function again.
}

/*! @brief Pass on focus tree parts from old owners to new owners
 *
 * @tparam     KeyType         32- or 64-bit unsigned integer
 * @param[in]  cstree          cornerstone leaf cell array (of the locally focused tree)
 * @param[in]  myRank          the executing rank
 * @param[in]  oldFocusStart   SFC assignment boundaries from previous iteration
 * @param[in]  oldFocusEnd
 * @param[in]  newFocusStart   new SFC assignment boundaries
 * @param[in]  newFocusEnd
 * @param[out] buffer          cell keys of parts of a remote rank's @p cstree for the newly assigned area of @p myRank
 *
 * When the assignment boundaries change, we let the previous owning rank pass on its focus tree of the part that it
 * lost to the new owning rank. Thanks to doing so we can guarantee that each rank always has the highest focus
 * tree resolution inside its focus of any rank: if rank a has focus SFC range F, then no other rank can have
 * tree cells in F that don't exist in rank a's focus tree.
 */
template<class KeyType>
void focusTransfer(gsl::span<const KeyType> cstree,
                   int myRank,
                   KeyType oldFocusStart,
                   KeyType oldFocusEnd,
                   KeyType newFocusStart,
                   KeyType newFocusEnd,
                   std::vector<KeyType>& buffer)
{
    constexpr int ownerTag = static_cast<int>(P2pTags::focusTransfer);

    std::vector<MPI_Request> sendRequests;

    if (oldFocusStart < newFocusStart)
    {
        // current rank lost range [oldFocusStart : newFocusStart] to rank below
        TreeNodeIndex start = findNodeAbove(cstree, oldFocusStart);
        TreeNodeIndex end   = findNodeAbove(cstree, newFocusStart);
        mpiSendAsync(cstree.data() + start, end - start, myRank - 1, ownerTag, sendRequests);
    }

    if (newFocusEnd < oldFocusEnd)
    {
        // current rank lost range [newFocusEnd : oldFocusEnd] to rank above
        TreeNodeIndex start = findNodeAbove(cstree, newFocusEnd);
        TreeNodeIndex end   = findNodeAbove(cstree, oldFocusEnd);
        int sendCount       = end - start;
        mpiSendAsync(cstree.data() + start, sendCount, myRank + 1, ownerTag, sendRequests);
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
