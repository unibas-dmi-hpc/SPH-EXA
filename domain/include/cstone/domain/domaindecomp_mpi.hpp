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

template<class T, class IndexType, class... Arrays>
void exchangeParticlesImpl(const SendList& sendList, int thisRank, std::size_t nParticlesAssigned,
                           IndexType inputOffset, IndexType outputOffset,
                           const IndexType* ordering, T* tempBuffer, Arrays... arrays)
{
    constexpr int domainExchangeTag = static_cast<int>(P2pTags::domainExchange);
    constexpr int numArrays = sizeof...(Arrays);
    std::array<T*, numArrays> sourceArrays{ (arrays + inputOffset)... };
    std::array<T*, numArrays> destinationArrays{ (arrays + outputOffset)... };

    int nRanks = int(sendList.size());

    std::vector<std::vector<T>> sendBuffers;
    sendBuffers.reserve(27); // 27 = maximum usual number of peers

    std::vector<MPI_Request> sendRequests;
    sendRequests.reserve(27);

    for (int destinationRank = 0; destinationRank < nRanks; ++destinationRank)
    {
        LocalParticleIndex sendCount = sendList[destinationRank].totalCount();
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

    // handle thisRank
    for (int arrayIndex = 0; arrayIndex < numArrays; ++arrayIndex)
    {
        // TODO: eliminate one copy by swapping the source with tempBuffer
        extractRange(sendList[thisRank], sourceArrays[arrayIndex], ordering, tempBuffer);
        std::copy(tempBuffer, tempBuffer + sendList[thisRank].totalCount(), destinationArrays[arrayIndex]);
    }

    std::size_t nParticlesPresent = sendList[thisRank].totalCount();

    std::vector<T> receiveBuffer;
    while (nParticlesPresent != nParticlesAssigned)
    {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, domainExchangeTag, MPI_COMM_WORLD, &status);
        int receiveRank = status.MPI_SOURCE;
        int receiveCountTotal;
        MPI_Get_count(&status, MpiType<T>{}, &receiveCountTotal);

        size_t receiveCount = receiveCountTotal / numArrays;
        if (nParticlesPresent + receiveCount > nParticlesAssigned)
        {
            throw std::runtime_error("Particle exchange: cannot receive more particles than assigned\n");
        }

        receiveBuffer.resize(receiveCountTotal);
        mpiRecvSync(receiveBuffer.data(), receiveCountTotal, receiveRank, domainExchangeTag, &status);

        for (int arrayIndex = 0; arrayIndex < numArrays; ++arrayIndex)
        {
            auto source = receiveBuffer.begin() + arrayIndex * receiveCount;
            std::copy(source, source + receiveCount, destinationArrays[arrayIndex] + nParticlesPresent);
        }

        nParticlesPresent += receiveCount;
    }

    if (not sendRequests.empty())
    {
        MPI_Status status[sendRequests.size()];
        MPI_Waitall(int(sendRequests.size()), sendRequests.data(), status);
    }

    // If this process is going to send messages with rank/tag combinations
    // already sent in this function, this can lead to messages being mixed up
    // on the receiver side. This happens e.g. with repeated consecutive calls of
    // this function. For this reason, a barrier is enacted here.
    // If there are no interfering messages going to be sent, it would be possible to
    // remove the barrier. But if that assumption turns out to be wrong, arising bugs
    // will be hard to detect.

    // MUST call MPI_Barrier or any other collective MPI operation that enforces synchronization
    // across all ranks before calling this function again.
    //MPI_Barrier(MPI_COMM_WORLD);
}

/*! @brief exchange array elements with other ranks according to the specified ranges
 *
 * @tparam T                      double, float or int
 * @tparam Arrays                 all std::vector<T>
 * @param[in] sendList            List of index ranges to be sent to each rank, indices
 *                                are valid w.r.t to arrays present on @p thisRank relative to the @p inputOffset.
 * @param[in] thisRank            Rank of the executing process
 * @param[in] nParticlesAssigned  New number of assigned particles for each array on @p thisRank.
 *                                This serves as the stop criterion for listening to incoming particles.
 * @param[in] inputOffset         Access arrays starting from @p inputOffset when extracting particles for sending
 * @param[in] outputOffset        Incoming particles will be added to their destination arrays starting from @p outputOffset
 * @param[in] ordering            Ordering through which to access arrays
 * @param[inout] arrays           T* pointers of identical sizes. The index range based exchange operations
 *                                performed are identical for each input array. Upon completion, arrays will
 *                                contain elements from the specified ranges from all ranks.
 *                                The order in which the incoming ranges are grouped is random.
 *
 *  Example: If sendList[ri] contains the range [upper, lower), all elements (arrays+inputOffset)[ordering[upper:lower]]
 *           will be sent to rank ri. At the destination ri, any assigned particles already present,
 *           are moved to their destination arrays, starting from @p outputOffset. The incoming elements to ri
 *           will be appended to the aforementioned elements.
 *           No information about incoming particles to @p thisRank is contained in the function arguments,
 *           only their total number @p nParticlesAssigned, which also includes any assigned particles
 *           already present on @p thisRank.
 *
 *  A note on the sizes of @p arrays:
 *  Let's set
 *      int nOldAssignment = the maximum of sendList[rank].lastNodeIdx(range) for any rank and range combination
 *  Then inputOffset + nOldAssignment is the upper bound for read accesses in @p arrays while sending.
 *  This is assuming that ordering.size() == nOldAssignment and that ordering[i] < nOldAssignment, which
 *  is what the Morten (re)order code produces.
 *  While writing during the receive phase, the highest index is outputOffset + nParticlesAssigned and it
 *  is checked that no writes occur past that location.
 *  In summary, the conditions to check in case of access violations / segmentation faults are for all arrays:
 *
 *      array.size() >= inputOffset + nOldAssignment
 *      array.size() >= outputOffset + nParticlesAssigned
 *      ordering.size() == nOldAssignment
 *      *std::max_element(begin(ordering), end(ordering)) == nOldAssignment - 1
 */
template<class T, class IndexType, class... Arrays>
void exchangeParticles(const SendList& sendList, Rank thisRank, IndexType nParticlesAssigned,
                       IndexType inputOffset, IndexType outputOffset, const IndexType* ordering, Arrays... arrays)
{
    IndexType nParticlesAlreadyPresent = sendList[thisRank].totalCount();
    std::vector<T> tempBuffer(nParticlesAlreadyPresent);
    exchangeParticlesImpl(sendList, thisRank, nParticlesAssigned, inputOffset, outputOffset,
                          ordering, tempBuffer.data(), arrays...);
}

/*! @brief exchange array elements with other ranks according to the specified ranges
 *
 * @tparam T                         double, float or int
 * @tparam Arrays                    all std::vector<T>
 * @param[in]    sendList            List of index ranges assigned to each rank, indices
 *                                   are valid w.r.t to arrays present on @p thisRank
 * @param[in]    thisRank            Rank of the executing process
 * @param[in]    nParticlesAssigned  Number of elements that each array will hold on @p thisRank after the exchange
 * @param[in]    ordering            Ordering through which to access arrays
 * @param[inout] arrays              T* pointers of identical sizes, the index range based exchange operations
 *                                   performed are identical for each input array. Upon completion, arrays will
 *                                   contain elements from the specified ranges from all ranks.
 *                                   The order in which the incoming ranges are grouped is random.
 *
 * See documentation of exchangeParticles with the full signature
 */
template<class T, class IndexType, class... Arrays>
void exchangeParticles(const SendList& sendList, Rank thisRank, IndexType nParticlesAssigned,
                       const IndexType* ordering, Arrays... arrays)
{
    exchangeParticles<T>(sendList, thisRank, nParticlesAssigned, IndexType(0), IndexType(0), ordering, arrays...);
}

} // namespace cstone
