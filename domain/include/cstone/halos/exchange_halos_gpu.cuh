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

#include "cstone/cuda/errorcheck.cuh"
#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/primitives/mpi_cuda.cuh"
#include "cstone/util/index_ranges.hpp"
#include "cstone/util/thrust_alloc.cuh"
#include "cstone/util/util.hpp"

#include "gather_scatter.cuh"

namespace cstone
{

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

    const size_t oldSendSize = sendScratchBuffer.size();
    reallocateDevice(sendScratchBuffer,
                     outgoingHalos.totalCount() * bytesPerElement / sizeof(typename DeviceVector::value_type), 1.01);

    char* sendBuffer = reinterpret_cast<char*>(thrust::raw_pointer_cast(sendScratchBuffer.data()));

    std::vector<MPI_Request> sendRequests;
    std::vector<std::vector<char, util::DefaultInitAdaptor<char>>> sendBuffers;

    int haloExchangeTag = static_cast<int>(P2pTags::haloExchange) + epoch;

    size_t numRanges = std::max(maxNumRanges(outgoingHalos), maxNumRanges(incomingHalos));
    IndexType* d_range;
    checkGpuErrors(cudaMalloc((void**)&d_range, 2 * numRanges * sizeof(IndexType)));
    IndexType* d_rangeScan = d_range + numRanges;

    char* sendPtr = sendBuffer;
    for (std::size_t destinationRank = 0; destinationRank < outgoingHalos.size(); ++destinationRank)
    {
        size_t sendCount = outgoingHalos[destinationRank].totalCount();
        if (sendCount == 0) continue;

        // compute indices to extract and upload to GPU
        auto [rangeOffsets, rangeScan] = createRanges(outgoingHalos[destinationRank]);
        checkGpuErrors(
            cudaMemcpy(d_range, rangeOffsets.data(), rangeOffsets.size() * sizeof(IndexType), cudaMemcpyHostToDevice));
        checkGpuErrors(
            cudaMemcpy(d_rangeScan, rangeScan.data(), rangeScan.size() * sizeof(IndexType), cudaMemcpyHostToDevice));

        util::array<size_t, numArrays> arrayByteOffsets = sendCount * elementSizes;
        std::exclusive_scan(arrayByteOffsets.begin(), arrayByteOffsets.end(), arrayByteOffsets.begin(), size_t(0));
        size_t sendBytes = sendCount * bytesPerElement;

        auto gatherArray = [sendPtr, sendCount, &data, &arrayByteOffsets, &elementSizes, d_range, d_rangeScan,
                            numRanges = rangeOffsets.size()](auto arrayIndex)
        {
            size_t outputOffset = arrayByteOffsets[arrayIndex];
            char* bufferPtr     = sendPtr + outputOffset;

            using ElementType = util::array<float, elementSizes[arrayIndex] / sizeof(float)>;
            gatherRanges(d_rangeScan, d_range, numRanges, reinterpret_cast<ElementType*>(data[arrayIndex]),
                         reinterpret_cast<ElementType*>(bufferPtr), sendCount);
        };

        for_each_tuple(gatherArray, indices);
        checkGpuErrors(cudaDeviceSynchronize());

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

    const size_t oldRecvSize = receiveScratchBuffer.size();
    reallocateDevice(receiveScratchBuffer, maxReceiveSize * bytesPerElement / sizeof(typename DeviceVector::value_type),
                     1.01);
    char* receiveBuffer = reinterpret_cast<char*>(thrust::raw_pointer_cast(receiveScratchBuffer.data()));

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
        checkGpuErrors(
            cudaMemcpy(d_range, rangeOffsets.data(), rangeOffsets.size() * sizeof(IndexType), cudaMemcpyHostToDevice));
        checkGpuErrors(
            cudaMemcpy(d_rangeScan, rangeScan.data(), rangeScan.size() * sizeof(IndexType), cudaMemcpyHostToDevice));

        auto scatterArray = [receiveBuffer, receiveCount, &data, &arrayByteOffsets, &elementSizes, d_range, d_rangeScan,
                             numRanges = rangeOffsets.size()](auto arrayIndex)
        {
            size_t outputOffset = arrayByteOffsets[arrayIndex];
            char* bufferPtr     = receiveBuffer + outputOffset;

            using ElementType = util::array<float, elementSizes[arrayIndex] / sizeof(float)>;
            scatterRanges(d_rangeScan, d_range, numRanges, reinterpret_cast<ElementType*>(data[arrayIndex]),
                          reinterpret_cast<ElementType*>(bufferPtr), receiveCount);
        };

        for_each_tuple(scatterArray, indices);
        checkGpuErrors(cudaDeviceSynchronize());

        numMessages--;
    }

    if (not sendRequests.empty())
    {
        MPI_Status status[sendRequests.size()];
        MPI_Waitall(int(sendRequests.size()), sendRequests.data(), status);
    }

    checkGpuErrors(cudaFree(d_range));
    reallocateDevice(sendScratchBuffer, oldSendSize, 1.01);
    reallocateDevice(receiveScratchBuffer, oldRecvSize, 1.01);

    // MUST call MPI_Barrier or any other collective MPI operation that enforces synchronization
    // across all ranks before calling this function again.
    // MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace cstone
