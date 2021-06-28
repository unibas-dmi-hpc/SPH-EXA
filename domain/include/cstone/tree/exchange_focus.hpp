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
void countRequestParticles(gsl::span<const KeyType> leaves, gsl::span<const unsigned> counts,
                           gsl::span<const KeyType> requestLeaves, gsl::span<unsigned> requestCounts)
{
    #pragma omp parallel for
    for (TreeNodeIndex i = 0; i < requestCounts.size(); ++i)
    {
        KeyType startKey = requestLeaves[i];
        KeyType endKey   = requestLeaves[i+1];

        TreeNodeIndex startIdx = findNodeBelow(leaves, startKey);
        TreeNodeIndex endIdx   = findNodeAbove(leaves, endKey);

        requestCounts[i] = std::accumulate(counts.begin() + startIdx, counts.begin() + endIdx, 0u);
    }
}

/*! @brief exchange particle counts with specified peer ranks
 *
 * @tparam KeyType                  32- or 64-bit unsigned integer
 * @param[in]  peerRanks            list of peer rank IDs
 * @param[in]  exchangeIndices      contains one range of indices of @p localLeaves to request counts
 *                                  for from each peer rank, length = same as @p peerRanks
 * @param[in]  localLeaves          cornerstone SFC key sequence of the locally (focused) tree
 *                                  of the executing rank
 * @param[out] localCounts          particle counts associated with @p localLeaves
 *                                  length(localCounts) = length(localLeaves) - 1
 * @param[-]   queryLeafBuffer      temp buffer for MPI p2p, length = same as @p localLeaves
 * @param[-]   queryCountBuffer     temp buffer for MPI p2p, length = same as @p localCounts
 *
 * Procedure on each rank:
 *  1. Send out the SFC keys for which it wants to get particle counts to peer ranks
 *  2. receive SFC keys from other peer ranks, count particles and send back the counts as answer
 *  3. receive answer with the counts for the requested keys
 */
template<class KeyType>
void exchangePeerCounts(gsl::span<const int> peerRanks, gsl::span<const IndexPair<TreeNodeIndex>> exchangeIndices,
                        gsl::span<const KeyType> localLeaves, gsl::span<unsigned> localCounts,
                        gsl::span<KeyType> queryLeafBuffer, gsl::span<unsigned> queryCountBuffer)

{
    std::vector<std::vector<unsigned>> sendBuffers;
    sendBuffers.reserve(peerRanks.size());

    std::vector<MPI_Request> sendRequests;
    for (size_t rankIndex = 0; rankIndex < peerRanks.size(); ++rankIndex)
    {
        int destinationRank = peerRanks[rankIndex];
        // +1 to include the upper key boundary for the last node
        TreeNodeIndex sendCount = exchangeIndices[rankIndex].count() + 1;
        mpiSendAsync(localLeaves.data() + exchangeIndices[rankIndex].start(), sendCount, destinationRank, 0, sendRequests);
    }

    size_t numMessages = peerRanks.size();
    while (numMessages > 0)
    {
        MPI_Status status;
        // receive SFC key sequence from remote rank, this defines the remote rank's node structure view of the local domain
        mpiRecvSync(queryLeafBuffer.data(), queryLeafBuffer.size(), MPI_ANY_SOURCE, 0, &status);
        int receiveRank = status.MPI_SOURCE;
        TreeNodeIndex numKeys;
        MPI_Get_count(&status, MpiType<KeyType>{}, &numKeys);

        // compute particle counts for the received node structure.
        // The number of nodes to count is one less the number of received SFC keys
        TreeNodeIndex numNodes = numKeys - 1;
        std::vector<unsigned> countBuffer(numNodes);
        countRequestParticles<KeyType>(localLeaves, localCounts, queryLeafBuffer.first(numKeys), countBuffer);

        // send back answer with the counts for the requested nodes
        mpiSendAsync(countBuffer.data(), numNodes, receiveRank, 1, sendRequests);
        sendBuffers.push_back(std::move(countBuffer));

        numMessages--;
    }

    numMessages = peerRanks.size();
    while (numMessages > 0)
    {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
        int receiveRank = status.MPI_SOURCE;

        size_t receiveRankIndex    = std::find(peerRanks.begin(), peerRanks.end(), receiveRank) - peerRanks.begin();
        TreeNodeIndex receiveCount = exchangeIndices[receiveRankIndex].count();
        mpiRecvSync(localCounts.data() + exchangeIndices[receiveRankIndex].start(), receiveCount, receiveRank, 1, &status);

        numMessages--;
    }

    MPI_Status status[sendRequests.size()];
    MPI_Waitall(int(sendRequests.size()), sendRequests.data(), status);

    MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace cstone
