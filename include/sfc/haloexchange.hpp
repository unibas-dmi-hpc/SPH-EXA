#pragma once

#include <vector>

#include "sfc/mpi_wrappers.hpp"

#include "sfc/layout.hpp"

namespace sphexa
{

template<class T, class...Arrays>
void haloexchange(const ArrayLayout& layout,
                  const std::vector<std::vector<int>>& incomingHalos,
                  const std::vector<std::vector<int>>& outgoingHalos,
                  int thisRank,
                  Arrays... arrays)
{
    constexpr int nArrays = sizeof...(Arrays);
    std::array<T*, nArrays> data{arrays...};

    std::vector<std::vector<T>> sendBuffers;
    std::vector<MPI_Request>    sendRequests;
    std::vector<int>            sendCounts(outgoingHalos.size());

    for (int destinationRank = 0; destinationRank < outgoingHalos.size(); ++destinationRank)
    {
        if (outgoingHalos[destinationRank].empty())
            continue;

        std::vector<T> buffer;
        for (int arrayIndex = 0; arrayIndex < nArrays; ++arrayIndex)
        {
            for (int ni = 0; ni < outgoingHalos[destinationRank].size(); ++ni)
            {
                int nodeIndex = outgoingHalos[destinationRank][ni];
                int offset    = layout.nodePosition(nodeIndex);
                int count     = layout.nodeCount(nodeIndex);

                std::copy(data[arrayIndex]+offset, data[arrayIndex]+offset+count,
                          std::back_inserter(buffer));
            }
        }

        sendCounts[destinationRank] = buffer.size();
        mpiSendAsync(&sendCounts[destinationRank], 1, destinationRank, 0, sendRequests);
        mpiSendAsync(buffer.data(), buffer.size(), destinationRank, 1, sendRequests);
        sendBuffers.push_back(std::move(buffer));
    }

    int nMessages = 0;
    for (int sourceRank = 0; sourceRank < incomingHalos.size(); ++sourceRank)
        if (!incomingHalos[sourceRank].empty())
            nMessages++;

    int maxReceiveSize = (layout.totalSize() - layout.localCount()) * nArrays;
    std::vector<T> receiveBuffer(maxReceiveSize);

    while (nMessages > 0)
    {
        MPI_Status status;
        int receiveCount;
        mpiRecvSync(&receiveCount, 1, MPI_ANY_SOURCE, 0, &status);

        int receiveRank = status.MPI_SOURCE;
        mpiRecvSync(receiveBuffer.data(), receiveCount, receiveRank, 1, &status);

        int countPerArray = receiveCount / nArrays;

        for (int arrayIndex = 0; arrayIndex < nArrays; ++arrayIndex)
        {
            int inputOffset = countPerArray * arrayIndex;
            for (int ni = 0; ni < incomingHalos[receiveRank].size(); ++ni)
            {
                int nodeIndex = incomingHalos[receiveRank][ni];
                int offset    = layout.nodePosition(nodeIndex);
                int count     = layout.nodeCount(nodeIndex);

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
        MPI_Waitall(sendRequests.size(), sendRequests.data(), status);
    }
}

} // namespace sphexa