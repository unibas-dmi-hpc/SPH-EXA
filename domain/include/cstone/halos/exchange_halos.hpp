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

#include <vector>

#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/util/index_ranges.hpp"

namespace cstone
{

template<class...Arrays>
void haloexchange(int epoch,
                  const SendList& incomingHalos,
                  const SendList& outgoingHalos,
                  Arrays... arrays)
{
    using T = std::common_type_t<std::decay_t<decltype(*arrays)>...>;
    using IndexType = SendManifest::IndexType;

    constexpr int nArrays = sizeof...(Arrays);
    std::array<T*, nArrays> data{arrays...};

    std::vector<std::vector<T>> sendBuffers;
    std::vector<MPI_Request>    sendRequests;

    int haloExchangeTag = static_cast<int>(P2pTags::haloExchange) + epoch;

    for (std::size_t destinationRank = 0; destinationRank < outgoingHalos.size(); ++destinationRank)
    {
        IndexType sendCount = outgoingHalos[destinationRank].totalCount();
        if (sendCount == 0)
            continue;

        std::vector<T> buffer(sendCount * nArrays);
        for (int arrayIndex = 0; arrayIndex < nArrays; ++arrayIndex)
        {
            IndexType outputOffset = sendCount * arrayIndex;
            for (std::size_t rangeIdx = 0; rangeIdx < outgoingHalos[destinationRank].nRanges(); ++rangeIdx)
            {
                IndexType lowerIndex = outgoingHalos[destinationRank].rangeStart(rangeIdx);
                IndexType upperIndex = outgoingHalos[destinationRank].rangeEnd(rangeIdx);

                std::copy(data[arrayIndex]+lowerIndex, data[arrayIndex]+upperIndex,
                          buffer.data() + outputOffset);
                outputOffset += upperIndex - lowerIndex;
            }
        }

        mpiSendAsync(buffer.data(), sendCount * nArrays, destinationRank, haloExchangeTag, sendRequests);
        sendBuffers.push_back(std::move(buffer));
    }

    int nMessages = 0;
    std::size_t maxReceiveSize = 0;
    for (std::size_t sourceRank = 0; sourceRank < incomingHalos.size(); ++sourceRank)
        if (incomingHalos[sourceRank].totalCount() > 0)
        {
            nMessages++;
            maxReceiveSize = std::max(maxReceiveSize, incomingHalos[sourceRank].totalCount());
        }

    std::vector<T> receiveBuffer(maxReceiveSize * nArrays);

    while (nMessages > 0)
    {
        MPI_Status status;
        mpiRecvSync(receiveBuffer.data(), receiveBuffer.size(), MPI_ANY_SOURCE, haloExchangeTag, &status);
        int receiveRank = status.MPI_SOURCE;
        size_t countPerArray = incomingHalos[receiveRank].totalCount();

        for (int arrayIndex = 0; arrayIndex < nArrays; ++arrayIndex)
        {
            size_t inputOffset = countPerArray * arrayIndex;
            for (std::size_t rangeIdx = 0; rangeIdx < incomingHalos[receiveRank].nRanges(); ++rangeIdx)
            {
                IndexType offset = incomingHalos[receiveRank].rangeStart(rangeIdx);
                size_t count     = incomingHalos[receiveRank].count(rangeIdx);

                std::copy(receiveBuffer.data() + inputOffset, receiveBuffer.data() + inputOffset + count,
                          data[arrayIndex]+offset);

                inputOffset += count;
            }
        }
        nMessages--;
    }

    if (not sendRequests.empty())
    {
        MPI_Status status[sendRequests.size()];
        MPI_Waitall(int(sendRequests.size()), sendRequests.data(), status);
    }

    // MUST call MPI_Barrier or any other collective MPI operation that enforces synchronization
    // across all ranks before calling this function again.
    //MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace cstone

