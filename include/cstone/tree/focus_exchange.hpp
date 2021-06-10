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
 * Overall procedure:
 *      1.  rank1 sends node structure (vector of SFC keys) to rank2. The node structure sent by rank1
 *          covers the assigned domain of rank2. The node structure cannot exceed the resolution of
 *          the local tree of rank2, this is guaranteed by the tree-build process, as long as all ranks
 *          use the same bucket size for the locally focused tree. Usually, rank1 requests the full resolution
 *          along the surface with rank2 and a lower resolution far a way from the surface.
 *
 *      2.  rank2 receives the the node structure, counts particles for each received node and sends back
 *          an answer with the particle counts per node.
 */

#pragma once

#include <vector>

#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/tree/octree.hpp"
#include "cstone/util/gsl-lite.hpp"

namespace cstone
{

template<class KeyType>
void countFocusParticles(gsl::span<const KeyType> leaves, gsl::span<const unsigned> counts,
                         gsl::span<const KeyType> requestLeaves, gsl::span<unsigned> requestCounts)
{
    #pragma omp parallel for
    for (TreeNodeIndex i = 0; i < requestCounts.size(); ++i)
    {
        KeyType startKey = requestLeaves[i];
        KeyType endKey   = requestLeaves[i+1];

        TreeNodeIndex startIdx = std::lower_bound(leaves.begin(), leaves.end(), startKey) - leaves.begin();
        TreeNodeIndex endIdx   = std::lower_bound(leaves.begin(), leaves.end(), endKey) - leaves.begin();

        requestCounts[i] = std::accumulate(counts.begin() + startIdx, counts.begin() + endIdx, 0u);
    }
}

bool exchangeConvergence(gsl::span<const int> peerRanks, int converged)
{
    if (peerRanks.empty()) { return converged; }

    std::vector<int> peerConvergence(peerRanks.size());

    std::vector<MPI_Request> sendRequests;
    for (int rankIndex = 0; rankIndex < int(peerRanks.size()); ++rankIndex)
    {
        int destinationRank = peerRanks[rankIndex];
        mpiSendAsync(&converged, 1, destinationRank, 3, sendRequests);
    }

    size_t numMessages = peerRanks.size();
    while (numMessages > 0)
    {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, 3, MPI_COMM_WORLD, &status);
        int receiveRank = status.MPI_SOURCE;

        size_t receiveRankIndex = std::find(peerRanks.begin(), peerRanks.end(), receiveRank) - peerRanks.begin();
        mpiRecvSync(peerConvergence.data() + receiveRankIndex, 1, receiveRank, 3, &status);

        numMessages--;
    }

    MPI_Status status[sendRequests.size()];
    MPI_Waitall(int(sendRequests.size()), sendRequests.data(), status);

    return converged && std::accumulate(peerConvergence.begin(), peerConvergence.end(), 0) == peerConvergence.size();
}

template<class KeyType>
void exchangeFocus(gsl::span<const int> peerRanks, gsl::span<const pair<TreeNodeIndex>> exchangeIndices,
                   gsl::span<const KeyType> focusLeaves, gsl::span<unsigned> focusCounts,
                   gsl::span<KeyType> queryLeafBuffer, gsl::span<unsigned> queryCountBuffer)

{
    std::vector<MPI_Request> sendRequests;
    for (int rankIndex = 0; rankIndex < int(peerRanks.size()); ++rankIndex)
    {
        int destinationRank = peerRanks[rankIndex];
        // +1 to include the upper key boundary for the last node
        TreeNodeIndex sendCount = exchangeIndices[rankIndex][1] - exchangeIndices[rankIndex][0] + 1;
        mpiSendAsync(focusLeaves.data() + exchangeIndices[rankIndex][0], sendCount, destinationRank, 0, sendRequests);
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
        countFocusParticles<KeyType>(focusLeaves, focusCounts, queryLeafBuffer.first(numKeys), queryCountBuffer.first(numNodes));

        // send back answer with the counts for the requested nodes
        //mpiSendAsync(queryCountBuffer.data(), numNodes, receiveRank, 1, sendRequests);
        MPI_Send(queryCountBuffer.data(), numNodes, MpiType<unsigned>{}, receiveRank, 1, MPI_COMM_WORLD);

        numMessages--;
    }

    numMessages = peerRanks.size();
    while (numMessages > 0)
    {
        MPI_Status status;
        mpiRecvSync(queryCountBuffer.data(), int(queryCountBuffer.size()), MPI_ANY_SOURCE, 1, &status);
        int receiveRank = status.MPI_SOURCE;

        size_t receiveRankIndex = std::find(peerRanks.begin(), peerRanks.end(), receiveRank) - peerRanks.begin();

        TreeNodeIndex receiveCount = exchangeIndices[receiveRankIndex][1] - exchangeIndices[receiveRankIndex][0];
        std::transform(queryCountBuffer.begin(), queryCountBuffer.begin() + receiveCount,
                       focusCounts.begin() + exchangeIndices[receiveRankIndex][0],
                       focusCounts.begin() + exchangeIndices[receiveRankIndex][0], std::plus<unsigned>{});

        numMessages--;
    }

    MPI_Status status[sendRequests.size()];
    MPI_Waitall(int(sendRequests.size()), sendRequests.data(), status);

    MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace cstone
