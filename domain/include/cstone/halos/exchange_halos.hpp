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

#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/util/index_ranges.hpp"

namespace cstone
{

template<class... Arrays>
void haloexchange(int epoch, const SendList& incomingHalos, const SendList& outgoingHalos, Arrays... arrays)
{
    using IndexType         = SendManifest::IndexType;
    constexpr int numArrays = sizeof...(Arrays);
    constexpr util::array<size_t, numArrays> elementSizes{sizeof(std::decay_t<decltype(*arrays)>)...};

    std::array<char*, numArrays> data{reinterpret_cast<char*>(arrays)...};

    std::vector<std::vector<char>> sendBuffers;
    std::vector<MPI_Request> sendRequests;

    int haloExchangeTag = static_cast<int>(P2pTags::haloExchange) + epoch;

    for (std::size_t destinationRank = 0; destinationRank < outgoingHalos.size(); ++destinationRank)
    {
        size_t sendCount = outgoingHalos[destinationRank].totalCount();
        if (sendCount == 0) continue;

        util::array<size_t, numArrays> arrayByteOffsets = sendCount * elementSizes;
        size_t totalBytes = std::accumulate(arrayByteOffsets.begin(), arrayByteOffsets.end(), size_t(0));
        std::exclusive_scan(arrayByteOffsets.begin(), arrayByteOffsets.end(), arrayByteOffsets.begin(), size_t(0));

        std::vector<char> buffer(totalBytes);
        for (int arrayIndex = 0; arrayIndex < numArrays; ++arrayIndex)
        {
            size_t outputOffset = arrayByteOffsets[arrayIndex];
            for (std::size_t rangeIdx = 0; rangeIdx < outgoingHalos[destinationRank].nRanges(); ++rangeIdx)
            {
                size_t lowerIndex = outgoingHalos[destinationRank].rangeStart(rangeIdx) * elementSizes[arrayIndex];
                size_t upperIndex = outgoingHalos[destinationRank].rangeEnd(rangeIdx) * elementSizes[arrayIndex];

                std::copy(data[arrayIndex] + lowerIndex, data[arrayIndex] + upperIndex, buffer.data() + outputOffset);
                outputOffset += upperIndex - lowerIndex;
            }
        }

        mpiSendAsync(buffer.data(), totalBytes, destinationRank, haloExchangeTag, sendRequests);
        sendBuffers.push_back(std::move(buffer));
    }

    int numMessages            = 0;
    std::size_t maxReceiveSize = 0;
    for (std::size_t sourceRank = 0; sourceRank < incomingHalos.size(); ++sourceRank)
        if (incomingHalos[sourceRank].totalCount() > 0)
        {
            numMessages++;
            maxReceiveSize = std::max(maxReceiveSize, incomingHalos[sourceRank].totalCount());
        }

    size_t bytesPerParticle = std::accumulate(elementSizes.begin(), elementSizes.end(), size_t(0));
    std::vector<char> receiveBuffer(maxReceiveSize * bytesPerParticle);

    while (numMessages > 0)
    {
        MPI_Status status;
        mpiRecvSync(receiveBuffer.data(), receiveBuffer.size(), MPI_ANY_SOURCE, haloExchangeTag, &status);
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

                std::copy(receiveBuffer.data() + inputOffset, receiveBuffer.data() + inputOffset + countBytes,
                          data[arrayIndex] + offset);

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

    // MUST call MPI_Barrier or any other collective MPI operation that enforces synchronization
    // across all ranks before calling this function again.
    // MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace cstone
