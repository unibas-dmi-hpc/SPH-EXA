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
 * @tparam     KeyType         32- or 64-bit unsigned integer
 * @param[in]  requestLeaves   query cornerstone SFC key sequence
 * @param[out] requestCounts   output counts for @p requestLeaves, length = length(requestLeaves) - 1
 * @param[in]  particleKeys    sorted SFC keys of local particles
 */
template<class KeyType>
void countRequestParticles(gsl::span<const KeyType> requestLeaves, gsl::span<unsigned> requestCounts,
                           gsl::span<const KeyType> particleKeys)
{
    #pragma omp parallel for
    for (size_t i = 0; i < requestCounts.size(); ++i)
    {
        requestCounts[i] = calculateNodeCount(requestLeaves.data(), i, particleKeys.data(),
                                              particleKeys.data() + particleKeys.size(),
                                              std::numeric_limits<unsigned>::max());
    }
}

/*! @brief count particles inside specified ranges of a cornerstone leaf tree
 *
 * @tparam KeyType             32- or 64-bit unsigned integer
 * @param[in]  leaves          cornerstone SFC key sequence,
 * @param[in]  counts          particle counts of @p leaves, length = length(leaves) - 1
 * @param[in]  requestLeaves   query cornerstone SFC key sequence
 * @param[out] requestCounts   output counts for @p requestLeaves, length = length(requestLeaves) - 1
 */
template<class KeyType>
void countRequestParticles(gsl::span<const KeyType> leaves, gsl::span<const unsigned> counts,
                           gsl::span<const KeyType> requestLeaves, gsl::span<unsigned> requestCounts)
{
    #pragma omp parallel for
    for (size_t i = 0; i < requestCounts.size(); ++i)
    {
        KeyType startKey = requestLeaves[i];
        KeyType endKey   = requestLeaves[i+1];

        size_t startIdx = findNodeBelow(leaves, startKey);
        size_t endIdx   = findNodeAbove(leaves, endKey);

        // Nodes in @p leaves must match the request keys exactly, otherwise counts are wrong.
        // If this assertion fails, it means that the local leaves/counts does not have the required
        // resolution to answer the incoming count request of [startKey:endKey] precisely.
        // In that case the overload above must be used which uses the particle keys directly to get the counts.
        assert(startKey == leaves[startIdx]);
        assert(endKey == leaves[endIdx]);

        requestCounts[i] = std::accumulate(counts.begin() + startIdx, counts.begin() + endIdx, 0u);
    }
}

template<class KeyType>
void exchangeTreelets(gsl::span<const int> peerRanks,
                      gsl::span<const IndexPair<TreeNodeIndex>> exchangeIndices,
                      gsl::span<const KeyType> localLeaves,
                      std::vector<std::vector<KeyType>>& peerTrees)

{
    constexpr int keyTag   = static_cast<int>(P2pTags::focusPeerCounts);
    size_t numPeers = peerRanks.size();

    std::vector<MPI_Request> sendRequests;
    sendRequests.reserve(numPeers);
    for (auto peer : peerRanks)
    {
        // +1 to include the upper key boundary for the last node
        TreeNodeIndex sendCount = exchangeIndices[peer].count() + 1;
        mpiSendAsync(localLeaves.data() + exchangeIndices[peer].start(), sendCount, peer, keyTag, sendRequests);
    }

    std::vector<MPI_Request> receiveRequests;
    receiveRequests.reserve(numPeers);
    int numMessages = numPeers;
    while (numMessages > 0)
    {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, keyTag, MPI_COMM_WORLD, &status);
        int receiveRank = status.MPI_SOURCE;
        TreeNodeIndex receiveSize;
        MPI_Get_count(&status, MpiType<KeyType>{}, &receiveSize);

        if (peerTrees[receiveRank].size() < receiveSize)
        {
            peerTrees[receiveRank].resize(receiveSize);
        }
        mpiRecvAsync(peerTrees[receiveRank].data(), receiveSize, receiveRank, keyTag, receiveRequests);

        numMessages--;
    }

    MPI_Status sendStatuses[numPeers];
    MPI_Waitall(int(numPeers), sendRequests.data(), sendStatuses);

    MPI_Status receiveStatuses[numPeers];
    MPI_Waitall(int(numPeers), receiveRequests.data(), receiveStatuses);
}

template<class KeyType>
void exchangeTreeletCounts(gsl::span<const int> peerRanks,
                           const std::vector<std::vector<KeyType>>& peerTrees,
                           gsl::span<const IndexPair<TreeNodeIndex>> exchangeIndices,
                           gsl::span<const KeyType> particleKeys,
                           gsl::span<unsigned> localCounts)
{
    size_t numPeers = peerRanks.size();
    std::vector<std::vector<unsigned>> sendBuffers;
    sendBuffers.reserve(numPeers);

    constexpr int countTag = static_cast<int>(P2pTags::focusPeerCounts) + 1;

    std::vector<MPI_Request> sendRequests;
    sendRequests.reserve(numPeers);
    for (auto peer : peerRanks)
    {
        // compute particle counts for the peer node structure.
        TreeNodeIndex numNodes = nNodes(peerTrees[peer]);
        std::vector<unsigned> countBuffer(numNodes);
        countRequestParticles<KeyType>(gsl::span(peerTrees[peer].data(), numNodes), countBuffer, particleKeys);

        // send back answer with the counts for the requested nodes
        mpiSendAsync(countBuffer.data(), numNodes, peer, countTag, sendRequests);
        sendBuffers.push_back(std::move(countBuffer));
    }

    int numMessages = peerRanks.size();
    while (numMessages > 0)
    {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, countTag, MPI_COMM_WORLD, &status);
        int receiveRank = status.MPI_SOURCE;

        TreeNodeIndex receiveCount = exchangeIndices[receiveRank].count();
        mpiRecvSync(localCounts.data() + exchangeIndices[receiveRank].start(), receiveCount, receiveRank, countTag, &status);

        numMessages--;
    }

    MPI_Status status[sendRequests.size()];
    MPI_Waitall(int(sendRequests.size()), sendRequests.data(), status);
}

/*! @brief exchange particle counts with specified peer ranks
 *
 * @tparam KeyType                  32- or 64-bit unsigned integer
 * @param[in]  peerRanks            list of peer rank IDs
 * @param[in]  exchangeIndices      contains one range of indices of @p localLeaves to request counts for each peer rank
 *                                  length = numRanks
 * @param[in]  particleKeys         sorted SFC keys of local particles
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
void exchangePeerCounts(gsl::span<const int> peerRanks, gsl::span<const IndexPair<TreeNodeIndex>> exchangeIndices,
                        gsl::span<const KeyType> particleKeys, gsl::span<const KeyType> localLeaves,
                        gsl::span<unsigned> localCounts)

{
    std::vector<std::vector<KeyType>> treelets(exchangeIndices.size());

    exchangeTreelets(peerRanks, exchangeIndices, localLeaves, treelets);
    exchangeTreeletCounts(peerRanks, treelets, exchangeIndices, particleKeys, localCounts);

    // MUST call MPI_Barrier or any other collective MPI operation that enforces synchronization
    // across all ranks before calling this function again.
    //MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace cstone
