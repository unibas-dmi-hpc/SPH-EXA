/*! @file
 * @brief  Halo particle exchange with MPI point-to-point communication
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <numeric>
#include <vector>

#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/domain/buffer_description.hpp"

namespace cstone
{

template<class... Arrays>
void haloexchange(int epoch, const RecvList& incomingHalos, const SendList& outgoingHalos, Arrays... arrays)
{
    using IndexType     = SendManifest::IndexType;
    int haloExchangeTag = static_cast<int>(P2pTags::haloExchange) + epoch;

    std::vector<std::vector<char>> sendBuffers;
    std::vector<MPI_Request> sendRequests;

    for (std::size_t destinationRank = 0; destinationRank < outgoingHalos.size(); ++destinationRank)
    {
        size_t sendCount = outgoingHalos[destinationRank].totalCount();
        if (sendCount == 0) continue;

        std::vector<char> buffer(util::computeByteOffsets(sendCount, 1, arrays...).back());

        auto packSendBuffer = [outHalos = outgoingHalos[destinationRank]](auto arrayPair)
        {
            for (std::size_t rangeIdx = 0; rangeIdx < outHalos.nRanges(); ++rangeIdx)
            {
                std::copy_n(arrayPair[0] + outHalos.rangeStart(rangeIdx), outHalos.count(rangeIdx), arrayPair[1]);
                arrayPair[1] += outHalos.count(rangeIdx);
            }
        };

        auto packTuple = util::packBufferPtrs<1>(buffer.data(), sendCount, arrays...);
        for_each_tuple(packSendBuffer, packTuple);

        mpiSendAsync(buffer.data(), buffer.size(), destinationRank, haloExchangeTag, sendRequests);
        sendBuffers.push_back(std::move(buffer));
    }

    int numMessages           = 0;
    LocalIndex maxReceiveSize = 0;
    for (const auto& incomingHalo : incomingHalos)
    {
        numMessages += int(incomingHalo.count() > 0);
        maxReceiveSize = std::max(maxReceiveSize, incomingHalo.count());
    }
    size_t maxReceiveBytes = util::computeByteOffsets(maxReceiveSize, 1, arrays...).back();

    std::vector<char> receiveBuffer(maxReceiveBytes);

    while (numMessages--)
    {
        MPI_Status status;
        mpiRecvSync(receiveBuffer.data(), receiveBuffer.size(), MPI_ANY_SOURCE, haloExchangeTag, &status);
        int receiveRank     = status.MPI_SOURCE;
        size_t receiveCount = incomingHalos[receiveRank].count();

        auto unpack = [inHalos = incomingHalos[receiveRank]](auto arrayPair)
        { std::copy_n(arrayPair[1], inHalos.count(), arrayPair[0] + inHalos.start()); };

        auto packTuple = util::packBufferPtrs<1>(receiveBuffer.data(), receiveCount, arrays...);
        for_each_tuple(unpack, packTuple);
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
