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

#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/primitives/mpi_cuda.cuh"
#include "cstone/util/index_ranges.hpp"
#include "cstone/util/thrust_noinit_alloc.cuh"

#include "gather_scatter.cuh"

namespace cstone
{

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

template<size_t... Is>
constexpr auto makeIntegralTuple(std::index_sequence<Is...>)
{
    return std::make_tuple(std::integral_constant<size_t, Is>{}...);
}

template<class DeviceVector, class... Arrays>
void haloExchangeGpu(int epoch,
                     const SendList& incomingHalos,
                     const SendList& outgoingHalos,
                     DeviceVector& sendScratchBuffer,
                     DeviceVector& receiveScratchBuffer,
                     Arrays... arrays)
{
    using IndexType         = SendManifest::IndexType;
    constexpr int numArrays = sizeof...(Arrays);
    constexpr util::array<size_t, numArrays> elementSizes{sizeof(std::decay_t<decltype(*arrays)>)...};
    const int bytesPerElement = std::accumulate(elementSizes.begin(), elementSizes.end(), 0);
    constexpr auto indices    = makeIntegralTuple(std::make_index_sequence<numArrays>{});

    std::array<char*, numArrays> data{reinterpret_cast<char*>(arrays)...};

    const size_t totalSendCount = sendCountSum(outgoingHalos);
    char* sendBuffer            = reinterpret_cast<char*>(thrust::raw_pointer_cast(sendScratchBuffer.data()));
    bool allocateSend =
        totalSendCount * bytesPerElement > sendScratchBuffer.size() * sizeof(typename DeviceVector::value_type);
    if (allocateSend)
    {
        //std::cout << "Allocating send buffer" << std::endl;
        cudaMalloc((void**)&sendBuffer, totalSendCount * bytesPerElement);
    }

    std::vector<MPI_Request> sendRequests;
    std::vector<std::vector<char, util::DefaultInitAdaptor<char>>> sendBuffers;

    int haloExchangeTag = static_cast<int>(P2pTags::haloExchange) + epoch;

    thrust::device_vector<IndexType, util::uninitialized_allocator<IndexType>> d_rangeOffsets;
    thrust::device_vector<IndexType, util::uninitialized_allocator<IndexType>> d_rangeScan;

    char* sendPtr = sendBuffer;
    for (std::size_t destinationRank = 0; destinationRank < outgoingHalos.size(); ++destinationRank)
    {
        size_t sendCount = outgoingHalos[destinationRank].totalCount();
        if (sendCount == 0) continue;

        // compute indices to extract and upload to GPU
        auto [rangeOffsets, rangeScan] = createRanges(outgoingHalos[destinationRank]);
        d_rangeOffsets                 = rangeOffsets;
        d_rangeScan                    = rangeScan;

        util::array<size_t, numArrays> arrayByteOffsets = sendCount * elementSizes;
        std::exclusive_scan(arrayByteOffsets.begin(), arrayByteOffsets.end(), arrayByteOffsets.begin(), size_t(0));
        size_t sendBytes = sendCount * bytesPerElement;

        auto gatherArray = [sendPtr, sendCount, &data, &arrayByteOffsets, &elementSizes, &d_rangeOffsets,
                            &d_rangeScan](auto arrayIndex)
        {
            size_t outputOffset = arrayByteOffsets[arrayIndex];
            char* bufferPtr     = sendPtr + outputOffset;

            using ElementType = util::array<float, elementSizes[arrayIndex] / sizeof(float)>;
            gatherRanges(thrust::raw_pointer_cast(d_rangeScan.data()), thrust::raw_pointer_cast(d_rangeOffsets.data()),
                         d_rangeOffsets.size(), reinterpret_cast<ElementType*>(data[arrayIndex]),
                         reinterpret_cast<ElementType*>(bufferPtr), sendCount);
        };

        for_each_tuple(gatherArray, indices);

        mpiSendGpuDirect(sendPtr, sendBytes, destinationRank, haloExchangeTag, sendRequests, sendBuffers);
        sendPtr += sendBytes;
    }

    int numMessages            = 0;
    std::size_t maxReceiveSize = 0;
    for (std::size_t sourceRank = 0; sourceRank < incomingHalos.size(); ++sourceRank)
    {
        if (incomingHalos[sourceRank].totalCount() > 0)
        {
            numMessages++;
            maxReceiveSize = std::max(maxReceiveSize, incomingHalos[sourceRank].totalCount());
        }
    }

    char* receiveBuffer = reinterpret_cast<char*>(thrust::raw_pointer_cast(sendScratchBuffer.data()));
    bool allocateReceive =
        maxReceiveSize * bytesPerElement > receiveScratchBuffer.size() * sizeof(typename DeviceVector::value_type);
    if (allocateReceive)
    {
        //std::cout << "Allocating receive buffer" << std::endl;
        cudaMalloc((void**)&receiveBuffer, bytesPerElement * maxReceiveSize);
    }

    while (numMessages > 0)
    {
        MPI_Status status;
        mpiRecvGpuDirect(receiveBuffer, maxReceiveSize * bytesPerElement, MPI_ANY_SOURCE, haloExchangeTag, &status);
        int receiveRank     = status.MPI_SOURCE;
        size_t receiveCount = incomingHalos[receiveRank].totalCount();

        util::array<size_t, numArrays> arrayByteOffsets = receiveCount * elementSizes;
        std::exclusive_scan(arrayByteOffsets.begin(), arrayByteOffsets.end(), arrayByteOffsets.begin(), size_t(0));

        // compute indices to extract and upload to GPU
        auto [rangeOffsets, rangeScan] = createRanges(incomingHalos[receiveRank]);
        d_rangeOffsets                 = rangeOffsets;
        d_rangeScan                    = rangeScan;

        auto scatterArray = [receiveBuffer, receiveCount, &data, &arrayByteOffsets, &elementSizes, &d_rangeOffsets,
                             &d_rangeScan](auto arrayIndex)
        {
            size_t outputOffset = arrayByteOffsets[arrayIndex];
            char* bufferPtr     = receiveBuffer + outputOffset;

            using ElementType = util::array<float, elementSizes[arrayIndex] / sizeof(float)>;
            scatterRanges(thrust::raw_pointer_cast(d_rangeScan.data()), thrust::raw_pointer_cast(d_rangeOffsets.data()),
                          d_rangeOffsets.size(), reinterpret_cast<ElementType*>(data[arrayIndex]),
                          reinterpret_cast<ElementType*>(bufferPtr), receiveCount);
        };

        for_each_tuple(scatterArray, indices);

        numMessages--;
    }

    if (not sendRequests.empty())
    {
        MPI_Status status[sendRequests.size()];
        MPI_Waitall(int(sendRequests.size()), sendRequests.data(), status);
    }

    if (allocateSend) { cudaFree(sendBuffer); }
    if (allocateReceive) { cudaFree(receiveBuffer); }

    // MUST call MPI_Barrier or any other collective MPI operation that enforces synchronization
    // across all ranks before calling this function again.
    // MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace cstone
