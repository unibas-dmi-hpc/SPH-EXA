#pragma once

#include "sfc/domaindecomp.hpp"

#include "mpi_wrappers.hpp"

namespace sphexa
{

/*! \brief exchange array elements with other ranks according to the specified ranges
 *
 * \tparam T                  double, float or int
 * \tparam Arrays             all std::vector<T>
 * \param sendList[in]        List of index ranges assigned to each rank, indices
 *                            are valid w.r.t to arrays present on \a thisRank
 * \param nParticlesAssigned  Number of elements that each array will hold on \a thisRank after the exchange
 * \param thisRank[in]        Rank of the executing process
 * \param ordering[in]        Ordering through which to access arrays
 * \param arrays[inout]       Arrays of identical sizes, the index range based exchange operations
 *                            performed are identical for each input array. Upon completion, arrays will
 *                            contain elements from the specified ranges from all ranks.
 *                            The order in which the incoming ranges are grouped is random.
 *
 *  Example: If sendList[ri] contains the range [upper, lower), all elements arrays[upper:lower] will be sent to rank ri.
 *           At the destination ri, the incoming elements will be appended to the corresponding arrays.
 *           No information about incoming particles to \a thisRank is contained in the function arguments,
 *           only their total number.
 */
template<class T, class... Arrays>
void exchangeParticles(const SendList& sendList, int nParticlesAssigned, int thisRank, const std::vector<int>& ordering, Arrays&... arrays)
{
    std::array<std::vector<T>*, sizeof...(Arrays)> data{ (&arrays)... };
    int nRanks = sendList.size();

    std::vector<std::vector<T>> sendBuffers;
    sendBuffers.reserve( data.size() * (nRanks-1));

    std::vector<MPI_Request> sendRequests;
    sendRequests.reserve( (2 + data.size()) * (nRanks-1));

    for (int destinationRank = 0; destinationRank < nRanks; ++destinationRank)
    {
        if (destinationRank == thisRank || sendList[destinationRank].totalCount() == 0) { continue; }

        mpiSendAsync(&thisRank, 1, destinationRank, 0, sendRequests);
        mpiSendAsync(&sendList[destinationRank].totalCount(), 1, destinationRank, 1, sendRequests);

        for (int arrayIndex = 0; arrayIndex < data.size(); ++arrayIndex)
        {
            auto arrayBuffer = createSendBuffer(sendList[destinationRank], *data[arrayIndex], ordering);
            mpiSendAsync(arrayBuffer.data(), arrayBuffer.size(), destinationRank, 2 + arrayIndex, sendRequests);
            sendBuffers.emplace_back(std::move(arrayBuffer));
        }
    }

    // handle thisRank
    for (int arrayIndex = 0; arrayIndex < data.size(); ++arrayIndex)
    {
        auto arrayBuffer = createSendBuffer(sendList[thisRank], *data[arrayIndex], ordering);
        std::copy(begin(arrayBuffer), end(arrayBuffer), data[arrayIndex]->begin());
    }

    unsigned nParticlesPresent  = sendList[thisRank].totalCount();
    for (auto array : data)
    {
        array->reserve(nParticlesAssigned);
        array->resize(nParticlesAssigned);
    }

    while (nParticlesPresent != nParticlesAssigned)
    {
        MPI_Status status[2 + data.size()];
        int receiveRank;
        std::size_t receiveRankCount;
        mpiRecvSync(&receiveRank, 1, MPI_ANY_SOURCE, 0, &status[0]);
        mpiRecvSync(&receiveRankCount, 1, receiveRank, 1, &status[1]);

        for (int arrayIndex = 0; arrayIndex < data.size(); ++arrayIndex)
        {
            mpiRecvSync(data[arrayIndex]->data() + nParticlesPresent, receiveRankCount, receiveRank, 2 + arrayIndex, &status[2 + arrayIndex]);
        }

        nParticlesPresent += receiveRankCount;
    }

    if (not sendRequests.empty())
    {
        MPI_Status status[sendRequests.size()];
        MPI_Waitall(sendRequests.size(), sendRequests.data(), status);
    }

    // If this process is going to send messages with rank/tag combinations
    // already sent in this function, this can lead to messages being mixed up
    // on the receiver side. This happens e.g. with repeated consecutive calls of
    // this function. For this reason, a barrier is enacted here.
    // If there are no interfering messages going to be sent, it would be possible to
    // remove the barrier. But if that assumption turns out to be wrong, arising bugs
    // will be hard to detect.
    MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace sphexa
