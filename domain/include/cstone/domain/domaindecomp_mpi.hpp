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
 * @param[inout] arrays           Pointers of different types but identical sizes. The index range based exchange
 *                                operations performed are identical for each input array. Upon completion, arrays will
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
std::tuple<LocalIndex, LocalIndex> exchangeParticles(const SendList& sendList,
                                                     int thisRank,
                                                     LocalIndex particleStart,
                                                     LocalIndex particleEnd,
                                                     LocalIndex arraySize,
                                                     LocalIndex numParticlesAssigned,
                                                     const LocalIndex* ordering,
                                                     Arrays... arrays)
{
    constexpr int domainExchangeTag = static_cast<int>(P2pTags::domainExchange);
    constexpr int numArrays         = sizeof...(Arrays);
    constexpr util::array<size_t, numArrays> elementSizes{sizeof(std::decay_t<decltype(*arrays)>)...};
    constexpr auto indices = makeIntegralTuple(std::make_index_sequence<numArrays>{});

    int numRanks = int(sendList.size());

    std::vector<std::vector<char>> sendBuffers;
    std::vector<MPI_Request> sendRequests;
    sendBuffers.reserve(27);
    sendRequests.reserve(27);

    std::array<char*, numArrays> sourceArrays{reinterpret_cast<char*>(arrays + particleStart)...};
    for (int destinationRank = 0; destinationRank < numRanks; ++destinationRank)
    {
        const auto& sends = sendList[destinationRank];
        size_t sendCount  = sends.totalCount();
        if (destinationRank == thisRank || sendCount == 0) { continue; }

        util::array<size_t, numArrays> arrayByteOffsets = sendCount * elementSizes;
        size_t totalBytes = std::accumulate(arrayByteOffsets.begin(), arrayByteOffsets.end(), size_t(0));
        std::exclusive_scan(arrayByteOffsets.begin(), arrayByteOffsets.end(), arrayByteOffsets.begin(), size_t(0));

        std::vector<char> sendBuffer(totalBytes);

        auto gatherArray = [sendPtr = sendBuffer.data(), sendCount, ordering, &sourceArrays, &arrayByteOffsets,
                            &elementSizes, rStart = sends.rangeStart(0)](auto arrayIndex)
        {
            size_t outputOffset = arrayByteOffsets[arrayIndex];
            char* bufferPtr     = sendPtr + outputOffset;

            using ElementType = util::array<float, elementSizes[arrayIndex] / sizeof(float)>;
            static_assert(elementSizes[arrayIndex] % sizeof(float) == 0, "elementSize must be a multiple of float");
            gather<LocalIndex>({ordering + rStart, sendCount}, reinterpret_cast<ElementType*>(sourceArrays[arrayIndex]),
                               reinterpret_cast<ElementType*>(bufferPtr));
        };
        for_each_tuple(gatherArray, indices);

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

    std::array<char*, numArrays> destinationArrays{reinterpret_cast<char*>(arrays + receiveStart)...};

    if (!fitHead && !fitTail && numIncoming > 0)
    {
        std::vector<char> tempBuffer(numParticlesPresent * *std::max_element(elementSizes.begin(), elementSizes.end()));

        auto gatherArray = [bufferPtr = tempBuffer.data(), ordering, &sourceArrays, &destinationArrays, &elementSizes,
                            rStart = sendList[thisRank].rangeStart(0), count = numParticlesPresent](auto index)
        {
            using ElementType = util::array<float, elementSizes[index] / sizeof(float)>;
            gather<LocalIndex>({ordering + rStart, count}, reinterpret_cast<ElementType*>(sourceArrays[index]),
                               reinterpret_cast<ElementType*>(bufferPtr));
            std::copy(bufferPtr, bufferPtr + count * elementSizes[index], destinationArrays[index]);
            destinationArrays[index] += count * elementSizes[index];
        };
        for_each_tuple(gatherArray, indices);
    }

    size_t bytesPerParticle = std::accumulate(elementSizes.begin(), elementSizes.end(), size_t(0));
    std::vector<char> receiveBuffer;
    while (numParticlesPresent != numParticlesAssigned)
    {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, domainExchangeTag, MPI_COMM_WORLD, &status);
        int receiveRank = status.MPI_SOURCE;
        int receiveCountBytes;
        MPI_Get_count(&status, MPI_CHAR, &receiveCountBytes);

        size_t receiveCount = receiveCountBytes / bytesPerParticle;
        assert(numParticlesPresent + receiveCount <= numParticlesAssigned);

        util::array<size_t, numArrays> arrayByteOffsets = receiveCount * elementSizes;
        std::exclusive_scan(arrayByteOffsets.begin(), arrayByteOffsets.end(), arrayByteOffsets.begin(), size_t(0));

        receiveBuffer.resize(receiveCountBytes);
        mpiRecvSync(receiveBuffer.data(), receiveCountBytes, receiveRank, domainExchangeTag, &status);

        for (int arrayIndex = 0; arrayIndex < numArrays; ++arrayIndex)
        {
            auto source = receiveBuffer.begin() + arrayByteOffsets[arrayIndex];
            std::copy(source, source + receiveCount * elementSizes[arrayIndex], destinationArrays[arrayIndex]);
            destinationArrays[arrayIndex] += receiveCount * elementSizes[arrayIndex];
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
    // MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace cstone
