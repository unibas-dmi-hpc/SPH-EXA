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
 * @brief  Exchange particles between different ranks to satisfy their assignments of the global octree
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "domaindecomp.hpp"

#include "cstone/primitives/mpi_wrappers.hpp"

namespace cstone
{

/*! @brief exchange array elements with other ranks according to the specified ranges
 *
 * @tparam Arrays                 pointers to particles buffers
 * @param[in] sendList            List of index ranges to be sent to each rank, indices
 *                                are valid w.r.t to arrays present on @p thisRank relative to @p particleStart.
 * @param[in] thisRank            Rank of the executing process
 * @param[in] particleStart       start index of locally owned particles prior to exchange
 * @param[in] particleEnd         end index of locally owned particles prior to exchange
 * @param[in] arraySize           size of @p arrays
 * @param[in] nParticlesAssigned  New number of assigned particles for each array on @p thisRank.
 * @param[in] ordering            Ordering through which to access arrays, valid w.r.t to [particleStart:particleEnd]
 * @param[inout] arrays           T* pointers of identical sizes. The index range based exchange operations
 *                                performed are identical for each input array. Upon completion, arrays will
 *                                contain elements from the specified ranges and ranks.
 *                                The order in which the incoming ranges are grouped is random.
 * @return                        (newStart, newEnd) tuple of indices delimiting the new range of assigned
 *                                particles post-exchange. Note: this range may contain left-over particles
 *                                from the previous assignment. Those can be removed with a subsequent call to
 *                                compactParticles().
 *
 *  Example: If sendList[ri] contains the range [upper, lower), all elements (arrays+inputOffset)[ordering[upper:lower]]
 *           will be sent to rank ri. At the destination ri, the incoming elements
 *           will be either prepended or appended to elements already present on ri.
 *           No information about incoming particles to @p thisRank is contained in the function arguments,
 *           only their total number @p nParticlesAssigned, which also includes any assigned particles
 *           already present on @p thisRank.
 */
template<class... Arrays>
std::tuple<LocalIndex, LocalIndex>
exchangeParticles(const SendList& sendList, int thisRank,
                                                     LocalIndex particleStart,
                                                     LocalIndex particleEnd,
                                                     LocalIndex arraySize,
                                                     LocalIndex numParticlesAssigned,
                  const LocalIndex* ordering, Arrays... arrays)
{
    using T = std::common_type_t<std::decay_t<decltype(*arrays)>...>;

    constexpr int domainExchangeTag = static_cast<int>(P2pTags::domainExchange);
    constexpr int numArrays = sizeof...(Arrays);
    int numRanks = int(sendList.size());

    std::vector<std::vector<T>> sendBuffers;
    std::vector<MPI_Request>    sendRequests;
    sendBuffers.reserve(27);
    sendRequests.reserve(27);

    std::array<T*, numArrays> sourceArrays{ (arrays + particleStart)... };
    for (int destinationRank = 0; destinationRank < numRanks; ++destinationRank)
    {
        LocalIndex sendCount = sendList[destinationRank].totalCount();
        if (destinationRank == thisRank || sendCount == 0) { continue; }

        std::vector<T> sendBuffer(numArrays * sendCount);
        for (int arrayIndex = 0; arrayIndex < numArrays; ++arrayIndex)
        {
            extractRange(sendList[destinationRank], sourceArrays[arrayIndex], ordering,
                         sendBuffer.data() + arrayIndex * sendCount);
        }
        mpiSendAsync(sendBuffer.data(), sendBuffer.size(), destinationRank, domainExchangeTag, sendRequests);
        sendBuffers.push_back(std::move(sendBuffer));
    }

    LocalIndex numParticlesPresent = sendList[thisRank].totalCount();
    LocalIndex numIncoming         = numParticlesAssigned - numParticlesPresent;

    bool fitHead = particleStart >= numIncoming;
    bool fitTail = arraySize - particleEnd >= numIncoming;

    LocalIndex receiveStart, newParticleStart, newParticleEnd;
    if (fitHead)
    {
        receiveStart     = particleStart - numIncoming;
        newParticleStart = particleStart - numIncoming;
        newParticleEnd   = particleEnd;
    }
    else if (fitTail)
    {
        receiveStart     = particleEnd;
        newParticleStart = particleStart;
        newParticleEnd   = particleEnd + numIncoming;
    }
    else
    {
        receiveStart     = 0;
        newParticleStart = 0;
        newParticleEnd   = numParticlesAssigned;
    }

    std::array<T*, numArrays> destinationArrays{ (arrays + receiveStart)... };

    if (!fitHead && !fitTail && numIncoming > 0)
    {
        std::vector<T> tempBuffer(numParticlesPresent);
        // handle thisRank
        for (int arrayIndex = 0; arrayIndex < numArrays; ++arrayIndex)
        {
            // make space by compacting already present particles
            extractRange(sendList[thisRank], sourceArrays[arrayIndex], ordering, tempBuffer.data());
            std::copy(begin(tempBuffer), end(tempBuffer), destinationArrays[arrayIndex]);
            destinationArrays[arrayIndex] += numParticlesPresent;
        }
    }

    std::vector<T> receiveBuffer;
    while (numParticlesPresent != numParticlesAssigned)
    {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, domainExchangeTag, MPI_COMM_WORLD, &status);
        int receiveRank = status.MPI_SOURCE;
        int receiveCountTotal;
        MPI_Get_count(&status, MpiType<T>{}, &receiveCountTotal);

        size_t receiveCount = receiveCountTotal / numArrays;
        assert(numParticlesPresent + receiveCount <= numParticlesAssigned);

        receiveBuffer.resize(receiveCountTotal);
        mpiRecvSync(receiveBuffer.data(), receiveCountTotal, receiveRank, domainExchangeTag, &status);

        for (int arrayIndex = 0; arrayIndex < numArrays; ++arrayIndex)
        {
            auto source = receiveBuffer.begin() + arrayIndex * receiveCount;
            std::copy(source, source + receiveCount, destinationArrays[arrayIndex]);
            destinationArrays[arrayIndex] += receiveCount;
        }

        numParticlesPresent += receiveCount;
    }

    if (not sendRequests.empty())
    {
        MPI_Status status[sendRequests.size()];
        MPI_Waitall(int(sendRequests.size()), sendRequests.data(), status);
    }

    return {newParticleStart, newParticleEnd};

    // If this process is going to send messages with rank/tag combinations
    // already sent in this function, this can lead to messages being mixed up
    // on the receiver side. This happens e.g. with repeated consecutive calls of
    // this function.

    // MUST call MPI_Barrier or any other collective MPI operation that enforces synchronization
    // across all ranks before calling this function again.
    //MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace cstone
