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

#include "domaindecomp.hpp"

namespace cstone
{

//! @brief copy the value of @a count to the start the provided GPU-buffer
void encodeSendCount(size_t count, char* sendPtr)
{
    checkGpuErrors(cudaMemcpy(sendPtr, &count, sizeof(size_t), cudaMemcpyHostToDevice));
}

//! @brief extract message length count from head of received GPU buffer and advance the buffer pointer by alignment
char* decodeSendCount(char* recvPtr, size_t* count, size_t alignment)
{
    checkGpuErrors(cudaMemcpy(count, recvPtr, sizeof(size_t), cudaMemcpyDeviceToHost));
    return recvPtr + alignment;
}

/*! @brief exchange array elements with other ranks according to the specified ranges
 *
 * @tparam Arrays                  pointers to particles buffers
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
 * @return                         (newStart, newEnd) tuple of indices delimiting the new range of assigned
 *                                 particles post-exchange. Note: this range may contain left-over particles
 *                                 from the previous assignment.
 *
 *  Example: If sendList[ri] contains the range [upper, lower), all elements (arrays+inputOffset)[ordering[upper:lower]]
 *           will be sent to rank ri. At the destination ri, the incoming elements
 *           will be either prepended or appended to elements already present on ri.
 *           No information about incoming particles to @p thisRank is contained in the function arguments,
 *           only their total number @p nParticlesAssigned, which also includes any assigned particles
 *           already present on @p thisRank.
 */
template<class DeviceVector, class... Arrays>
std::tuple<LocalIndex, LocalIndex> exchangeParticlesGpu(const SendList& sendList,
                                                        int thisRank,
                                                        LocalIndex particleStart,
                                                        LocalIndex particleEnd,
                                                        LocalIndex arraySize,
                                                        LocalIndex numParticlesAssigned,
                                                        DeviceVector& sendScratchBuffer,
                                                        DeviceVector& receiveScratchBuffer,
                                                        const LocalIndex* ordering,
                                                        Arrays... arrays)
{
    constexpr int domainExchangeTag = static_cast<int>(P2pTags::domainExchange);
    constexpr int numArrays         = sizeof...(Arrays);
    constexpr auto indices          = makeIntegralTuple(std::make_index_sequence<numArrays>{});
    constexpr size_t alignment      = 128;
    constexpr util::array<size_t, numArrays> elementSizes{sizeof(std::decay_t<decltype(*arrays)>)...};

    using TransferType = uint64_t;
    static_assert(alignment % sizeof(TransferType) == 0);

    size_t totalSendBytes    = computeTotalSendBytes(sendList, elementSizes, thisRank, alignment);
    const size_t oldSendSize = reallocateBytes(sendScratchBuffer, totalSendBytes);
    char* const sendBuffer   = reinterpret_cast<char*>(rawPtr(sendScratchBuffer));

    // Not used if GPU-direct is ON
    std::vector<std::vector<TransferType, util::DefaultInitAdaptor<TransferType>>> sendBuffers;
    std::vector<MPI_Request> sendRequests;

    std::array<char*, numArrays> sourceArrays{reinterpret_cast<char*>(arrays + particleStart)...};
    char* sendPtr = sendBuffer;
    for (int destinationRank = 0; destinationRank < int(sendList.size()); ++destinationRank)
    {
        const auto& sends = sendList[destinationRank];
        size_t sendCount  = sends.totalCount();
        if (destinationRank == thisRank || sendCount == 0) { continue; }

        encodeSendCount(sendCount, sendPtr);
        auto byteOffsets = computeByteOffsets(sendCount, elementSizes, alignment);
        auto gatherArray = [sendPtr, sendCount, ordering, &sourceArrays, &byteOffsets, &elementSizes,
                            rStart = sends.rangeStart(0)](auto arrayIndex)
        {
            size_t outputOffset = byteOffsets[arrayIndex];
            char* bufferPtr     = sendPtr + alignment + outputOffset;
            using ElementType   = util::array<float, elementSizes[arrayIndex] / sizeof(float)>;
            gatherGpu(ordering + rStart, sendCount, reinterpret_cast<ElementType*>(sourceArrays[arrayIndex]),
                      reinterpret_cast<ElementType*>(bufferPtr));
        };
        for_each_tuple(gatherArray, indices);
        checkGpuErrors(cudaDeviceSynchronize());

        mpiSendGpuDirect(reinterpret_cast<TransferType*>(sendPtr),
                         (alignment + byteOffsets.back()) / sizeof(TransferType), destinationRank, domainExchangeTag,
                         sendRequests, sendBuffers);
        sendPtr += alignment + byteOffsets.back();
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
    const size_t oldRecvSize = receiveScratchBuffer.size();

    if (!fitHead && !fitTail && numIncoming > 0 && numParticlesPresent > 0)
    {
        std::size_t requiredBytes = numParticlesPresent * *std::max_element(elementSizes.begin(), elementSizes.end());
        reallocateBytes(receiveScratchBuffer, requiredBytes);
        char* bufferPtr = reinterpret_cast<char*>(rawPtr(receiveScratchBuffer));

        auto gatherArray = [bufferPtr, ordering, &sourceArrays, &destinationArrays, &elementSizes,
                            rStart = sendList[thisRank].rangeStart(0), count = numParticlesPresent](auto index)
        {
            using ElementType = util::array<float, elementSizes[index] / sizeof(float)>;
            gatherGpu(ordering + rStart, count, reinterpret_cast<ElementType*>(sourceArrays[index]),
                      reinterpret_cast<ElementType*>(bufferPtr));
            checkGpuErrors(
                cudaMemcpy(destinationArrays[index], bufferPtr, count * elementSizes[index], cudaMemcpyDeviceToDevice));
            destinationArrays[index] += count * elementSizes[index];
        };
        for_each_tuple(gatherArray, indices);
        checkGpuErrors(cudaDeviceSynchronize());
    }

    while (numParticlesPresent != numParticlesAssigned)
    {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, domainExchangeTag, MPI_COMM_WORLD, &status);
        int receiveRank = status.MPI_SOURCE;
        int receiveCountTransfer;
        MPI_Get_count(&status, MpiType<TransferType>{}, &receiveCountTransfer);

        size_t receiveCountBytes = receiveCountTransfer * sizeof(TransferType);
        reallocateBytes(receiveScratchBuffer, receiveCountBytes);
        char* receiveBuffer = reinterpret_cast<char*>(rawPtr(receiveScratchBuffer));
        mpiRecvGpuDirect(reinterpret_cast<TransferType*>(receiveBuffer), receiveCountTransfer, receiveRank,
                         domainExchangeTag, &status);

        size_t receiveCount;
        receiveBuffer = decodeSendCount(receiveBuffer, &receiveCount, alignment);
        assert(numParticlesPresent + receiveCount <= numParticlesAssigned);

        auto byteOffsets = computeByteOffsets(receiveCount, elementSizes, alignment);
        for (int arrayIndex = 0; arrayIndex < numArrays; ++arrayIndex)
        {
            char* source = receiveBuffer + byteOffsets[arrayIndex];
            checkGpuErrors(cudaMemcpy(destinationArrays[arrayIndex], source, receiveCount * elementSizes[arrayIndex],
                                      cudaMemcpyDeviceToDevice));
            destinationArrays[arrayIndex] += receiveCount * elementSizes[arrayIndex];
        }
        checkGpuErrors(cudaDeviceSynchronize());

        numParticlesPresent += receiveCount;
    }

    if (not sendRequests.empty())
    {
        MPI_Status status[sendRequests.size()];
        MPI_Waitall(int(sendRequests.size()), sendRequests.data(), status);
    }

    reallocateDevice(sendScratchBuffer, oldSendSize, 1.01);
    reallocateDevice(receiveScratchBuffer, oldRecvSize, 1.01);

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
