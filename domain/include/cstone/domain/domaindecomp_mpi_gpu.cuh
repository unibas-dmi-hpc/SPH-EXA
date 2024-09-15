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

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/cuda/errorcheck.cuh"
#include "cstone/primitives/mpi_cuda.cuh"
#include "cstone/primitives/primitives_gpu.h"
#include "cstone/util/reallocate.hpp"

#include "buffer_description.hpp"

namespace cstone
{

//! @brief copy the value of @a count to the start the provided GPU-buffer
inline void encodeSendCount(size_t count, char* sendPtr)
{
    checkGpuErrors(cudaMemcpy(sendPtr, &count, sizeof(size_t), cudaMemcpyHostToDevice));
}

//! @brief extract message length count from head of received GPU buffer and advance the buffer pointer by alignment
inline char* decodeSendCount(char* recvPtr, size_t* count, size_t alignment)
{
    checkGpuErrors(cudaMemcpy(count, recvPtr, sizeof(size_t), cudaMemcpyDeviceToHost));
    return recvPtr + alignment;
}

/*! @brief exchange array elements with other ranks according to the specified ranges
 *
 * @tparam Arrays                  pointers to particles buffers
 * @param[in] epoch                MPI tag offset to avoid mix-ups of message from consecutive function calls
 * @param[in] receiveLog           List of received messages in previous calls to replicate resulting buffer layout
 * @param[in] sendList             List of index ranges to be sent to each rank, indices
 *                                 are valid w.r.t to arrays present on @p thisRank relative to @p particleStart.
 * @param[in] thisRank             Rank of the executing process
 * @param[in] particleStart        start index of locally owned particles prior to exchange
 * @param[in] particleEnd          end index of locally owned particles prior to exchange
 * @param[in] arraySize            size of @p arrays
 * @param[in] numParticlesAssigned New number of assigned particles for each array on @p thisRank.
 * @param[-]  sendScratchBuffer    resizable device vector for temporary usage
 * @param[-]  sendReceiveBuffer    resizable device vector for temporary usage
 * @param[in] ordering             Ordering to access arrays, valid w.r.t to [particleStart:particleEnd], ON DEVICE.
 * @param[inout] arrays            Pointers of different types but identical sizes. The index range based exchange
 *                                 operations performed are identical for each input array. Upon completion, arrays will
 *                                 contain elements from the specified ranges and ranks.
 *                                 The order in which the incoming ranges are grouped is random. ON DEVICE.
 *
 *  Example: If sendList[ri] contains the range [upper, lower), all elements (arrays+inputOffset)[ordering[upper:lower]]
 *           will be sent to rank ri. At the destination ri, the incoming elements
 *           will be either prepended or appended to elements already present on ri.
 *           No information about incoming particles to @p thisRank is contained in the function arguments,
 *           only their total number @p nParticlesAssigned, which also includes any assigned particles
 *           already present on @p thisRank.
 */
template<class DeviceVector, class... Arrays>
void exchangeParticlesGpu(int epoch,
                          ExchangeLog& receiveLog,
                          const SendRanges& sends,
                          int thisRank,
                          BufferDescription bufDesc,
                          LocalIndex numParticlesAssigned,
                          DeviceVector& sendScratchBuffer,
                          DeviceVector& receiveScratchBuffer,
                          const LocalIndex* ordering,
                          Arrays... arrays)
{
    using TransferType          = uint64_t;
    constexpr int alignment     = 128;
    constexpr int headerBytes   = round_up(sizeof(uint64_t), alignment);
    const float allocGrowthRate = 1.05;
    static_assert(alignment % sizeof(TransferType) == 0);
    bool record  = receiveLog.empty();
    int domExTag = static_cast<int>(P2pTags::domainExchange) + epoch;

    size_t totalSendBytes    = computeTotalSendBytes<alignment>(sends, thisRank, headerBytes, arrays...);
    const size_t oldSendSize = reallocateBytes(sendScratchBuffer, totalSendBytes, allocGrowthRate);
    char* const sendBuffer   = reinterpret_cast<char*>(rawPtr(sendScratchBuffer));

    // Not used if GPU-direct is ON
    std::vector<std::vector<TransferType, util::DefaultInitAdaptor<TransferType>>> sendBuffers;
    std::vector<MPI_Request> sendRequests;

    char* sendPtr = sendBuffer;
    for (int destinationRank = 0; destinationRank < sends.numRanks(); ++destinationRank)
    {
        size_t sendCount = sends.count(destinationRank);
        if (destinationRank == thisRank || sendCount == 0) { continue; }
        size_t sendStart = sends[destinationRank];

        encodeSendCount(sendCount, sendPtr);
        size_t numBytes = headerBytes + packArrays<alignment>(gatherGpuL, ordering + sendStart, sendCount,
                                                              sendPtr + headerBytes, arrays + bufDesc.start...);
        checkGpuErrors(cudaDeviceSynchronize());
        mpiSendGpuDirect(sendPtr, numBytes, destinationRank, domExTag, sendRequests, sendBuffers);
        sendPtr += numBytes;
    }

    LocalIndex numParticlesPresent = sends.count(thisRank);
    LocalIndex receiveStart        = domain_exchange::receiveStart(bufDesc, numParticlesPresent, numParticlesAssigned);
    LocalIndex receiveEnd          = receiveStart + numParticlesAssigned - numParticlesPresent;

    const size_t oldRecvSize = receiveScratchBuffer.size();
    while (receiveStart != receiveEnd)
    {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, domExTag, MPI_COMM_WORLD, &status);
        int receiveRank = status.MPI_SOURCE;
        int receiveCountTransfer;
        MPI_Get_count(&status, MpiType<TransferType>{}, &receiveCountTransfer);

        size_t receiveCountBytes = receiveCountTransfer * sizeof(TransferType);
        reallocateBytes(receiveScratchBuffer, receiveCountBytes, allocGrowthRate);
        char* receiveBuffer = reinterpret_cast<char*>(rawPtr(receiveScratchBuffer));
        mpiRecvGpuDirect(reinterpret_cast<TransferType*>(receiveBuffer), receiveCountTransfer, receiveRank, domExTag,
                         &status);

        size_t receiveCount;
        receiveBuffer = decodeSendCount(receiveBuffer, &receiveCount, alignment);
        assert(receiveStart + receiveCount <= receiveEnd);

        LocalIndex receiveLocation = receiveStart;
        if (record) { receiveLog.addExchange(receiveRank, receiveStart); }
        else { receiveLocation = receiveLog.lookup(receiveRank); }

        auto packTuple = util::packBufferPtrs<alignment>(receiveBuffer, receiveCount, (arrays + receiveLocation)...);
        auto scatterRanges = [receiveCount](auto arrayPair)
        {
            checkGpuErrors(cudaMemcpy(arrayPair[0], arrayPair[1],
                                      receiveCount * sizeof(std::decay_t<decltype(*arrayPair[0])>),
                                      cudaMemcpyDeviceToDevice));
        };
        util::for_each_tuple(scatterRanges, packTuple);
        checkGpuErrors(cudaDeviceSynchronize());

        receiveStart += receiveCount;
    }

    if (not sendRequests.empty())
    {
        MPI_Status status[sendRequests.size()];
        MPI_Waitall(int(sendRequests.size()), sendRequests.data(), status);
    }

    reallocate(sendScratchBuffer, oldSendSize, 1.01);
    reallocate(receiveScratchBuffer, oldRecvSize, 1.01);

    // If this process is going to send messages with rank/tag combinations
    // already sent in this function, this can lead to messages being mixed up
    // on the receiver side. This happens e.g. with repeated consecutive calls of
    // this function.

    // MUST call MPI_Barrier or any other collective MPI operation that enforces synchronization
    // across all ranks before calling this function again.
    // MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace cstone
