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
 * @brief  Halo particle exchange with MPI point-to-point communication
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <numeric>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/util/index_ranges.hpp"

namespace cstone
{

//template<class T, class IndexType>
//__global__ void gatherSend(const IndexType* rangeScan,
//                           const IndexType* rangeOffsets,
//                           int numRanges,
//                           const T* src,
//                           T* buffer,
//                           IndexType bufferSize)
//{
//    IndexType tid = blockIdx.x * blockDim.x + threadIdx.x;
//    if (tid < bufferSize)
//    {
//        IndexType rangeIdx = stl::upper_bound(rangeScan, rangeScan + numRanges, tid) - rangeScan - 1;
//
//        IndexType srcIdx = rangeOffsets[rangeIdx] + tid - rangeScan[rangeIdx];
//        buffer[tid]      = src[srcIdx];
//    }
//}

auto createRanges(const SendManifest& ranges)
{
    using IndexType = SendManifest::IndexType;
    std::vector<IndexType> offsets(ranges.nRanges());
    std::vector<IndexType> scan(ranges.nRanges());

    for (IndexType i = 0; i < ranges.nRanges(); ++i)
    {
        offsets[i] = ranges.rangeStart(i);
        scan[i]    = ranges.count(i);
    }

    std::exclusive_scan(scan.begin(), scan.end(), scan.begin(), IndexType(0));
    return std::make_tuple(std::move(offsets), std::move(scan));
}

size_t sendCountSum(const SendList& outgoingHalos)
{
    size_t sendCount = 0;
    for (std::size_t destinationRank = 0; destinationRank < outgoingHalos.size(); ++destinationRank)
    {
        sendCount += outgoingHalos[destinationRank].totalCount();
    }
    return sendCount;
}

template<class... Arrays>
void haloExchangeGpu(int epoch, const SendList& incomingHalos, const SendList& outgoingHalos, Arrays... arrays)
{
    using IndexType         = SendManifest::IndexType;
    constexpr int numArrays = sizeof...(Arrays);
    constexpr util::array<size_t, numArrays> elementSizes{sizeof(std::decay_t<decltype(*arrays)>)...};
    int bytesPerElement = std::accumulate(elementSizes.begin(), elementSizes.end(), 0);

    std::array<char*, numArrays> data{reinterpret_cast<char*>(arrays)...};

    size_t totalSendCount = sendCountSum(outgoingHalos);
    char* sendPtr;
    cudaMalloc((void**)&sendPtr, bytesPerElement * totalSendCount);
    std::vector<MPI_Request> sendRequests;

    int haloExchangeTag = static_cast<int>(P2pTags::haloExchange) + epoch;

    //thrust::device_vector<IndexType> d_rangeOffsets;
    //thrust::device_vector<IndexType> d_rangeScan;

    for (std::size_t destinationRank = 0; destinationRank < outgoingHalos.size(); ++destinationRank)
    {
        size_t sendCount = outgoingHalos[destinationRank].totalCount();
        if (sendCount == 0) continue;

        // compute indices to extract and upload to GPU
        //auto [rangeOffsets, rangeScan] = createRanges(outgoingHalos[destinationRank]);
        //d_rangeOffsets                 = rangeOffsets;
        //d_rangeScan                    = rangeScan;

        util::array<size_t, numArrays> arrayByteOffsets = sendCount * elementSizes;
        std::exclusive_scan(arrayByteOffsets.begin(), arrayByteOffsets.end(), arrayByteOffsets.begin(), size_t(0));
        size_t sendBytes = sendCount * bytesPerElement;

        for (int arrayIndex = 0; arrayIndex < numArrays; ++arrayIndex)
        {
            size_t outputOffset = arrayByteOffsets[arrayIndex];
            //char* bufferPtr     = thrust::raw_pointer_cast(buffer.data()) + outputOffset;

            //using ElementType = util::array<float, elementSizes[0] / sizeof(float)>;
            //int numThreads    = 256;
            //int numBlocks     = iceil(sendCount, numThreads);
            //gatherSend<<<numBlocks, numThreads>>>(thrust::raw_pointer_cast(d_rangeScan.data()),
            //                                      thrust::raw_pointer_cast(d_rangeOffsets.data()), rangeOffsets.size(),
            //                                      reinterpret_cast<ElementType*>(data[arrayIndex]),
            //                                      reinterpret_cast<ElementType*>(bufferPtr), sendCount);

            for (std::size_t rangeIdx = 0; rangeIdx < outgoingHalos[destinationRank].nRanges(); ++rangeIdx)
            {
                size_t lowerIndex = outgoingHalos[destinationRank].rangeStart(rangeIdx) * elementSizes[arrayIndex];
                size_t upperIndex = outgoingHalos[destinationRank].rangeEnd(rangeIdx) * elementSizes[arrayIndex];

                cudaMemcpy(sendPtr + outputOffset, data[arrayIndex] + lowerIndex, upperIndex - lowerIndex,
                           cudaMemcpyDeviceToDevice);
                outputOffset += upperIndex - lowerIndex;
            }
        }

        mpiSendAsync(sendPtr, sendBytes, destinationRank, haloExchangeTag, sendRequests);
        sendPtr += sendBytes;
    }

    int numMessages            = 0;
    std::size_t maxReceiveSize = 0;
    for (std::size_t sourceRank = 0; sourceRank < incomingHalos.size(); ++sourceRank)
        if (incomingHalos[sourceRank].totalCount() > 0)
        {
            numMessages++;
            maxReceiveSize = std::max(maxReceiveSize, incomingHalos[sourceRank].totalCount());
        }

    char* receiveBuffer;
    cudaMalloc((void**)&receiveBuffer, bytesPerElement * maxReceiveSize);

    while (numMessages > 0)
    {
        MPI_Status status;
        mpiRecvSync(receiveBuffer, maxReceiveSize, MPI_ANY_SOURCE, haloExchangeTag, &status);
        int receiveRank     = status.MPI_SOURCE;
        size_t receiveCount = incomingHalos[receiveRank].totalCount();

        util::array<size_t, numArrays> arrayByteOffsets = receiveCount * elementSizes;
        std::exclusive_scan(arrayByteOffsets.begin(), arrayByteOffsets.end(), arrayByteOffsets.begin(), size_t(0));

        for (int arrayIndex = 0; arrayIndex < numArrays; ++arrayIndex)
        {
            size_t inputOffset = arrayByteOffsets[arrayIndex];
            for (std::size_t rangeIdx = 0; rangeIdx < incomingHalos[receiveRank].nRanges(); ++rangeIdx)
            {
                IndexType offset  = incomingHalos[receiveRank].rangeStart(rangeIdx) * elementSizes[arrayIndex];
                size_t countBytes = incomingHalos[receiveRank].count(rangeIdx) * elementSizes[arrayIndex];

                cudaMemcpy(data[arrayIndex] + offset, receiveBuffer + inputOffset, countBytes,
                           cudaMemcpyDeviceToDevice);

                inputOffset += countBytes;
            }
        }
        numMessages--;
    }

    if (not sendRequests.empty())
    {
        MPI_Status status[sendRequests.size()];
        MPI_Waitall(int(sendRequests.size()), sendRequests.data(), status);
    }

    cudaFree(sendPtr);
    cudaFree(receiveBuffer);

    // MUST call MPI_Barrier or any other collective MPI operation that enforces synchronization
    // across all ranks before calling this function again.
    // MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace cstone
