#pragma once

#include "sfc/domaindecomp.hpp"

#include "mpi_wrappers.hpp"

namespace sphexa
{

template<class T, class... Arrays>
void exchangeParticlesImpl(const SendList& sendList, int thisRank, int nParticlesAssigned,
                           int inputOffset, int outputOffset,
                           const int* ordering, T* tempBuffer, Arrays&... arrays)
{
    //std::array<std::vector<T>*, sizeof...(Arrays)> data{ (&arrays)... };
    constexpr int nArrays = sizeof...(Arrays);
    std::array<T*, nArrays> sourceArrays{ (arrays.data() + inputOffset)... };
    std::array<T*, nArrays> destinationArrays{ (arrays.data() + outputOffset)... };

    int nRanks = sendList.size();

    std::vector<std::vector<T>> sendBuffers;
    sendBuffers.reserve(nArrays * (nRanks-1));

    std::vector<MPI_Request> sendRequests;
    sendRequests.reserve( (2 + nArrays) * (nRanks-1));

    for (int destinationRank = 0; destinationRank < nRanks; ++destinationRank)
    {
        if (destinationRank == thisRank || sendList[destinationRank].totalCount() == 0) { continue; }

        mpiSendAsync(&thisRank, 1, destinationRank, 0, sendRequests);
        mpiSendAsync(&sendList[destinationRank].totalCount(), 1, destinationRank, 1, sendRequests);

        for (int arrayIndex = 0; arrayIndex < nArrays; ++arrayIndex)
        {
            auto arrayBuffer = createSendBuffer(sendList[destinationRank], sourceArrays[arrayIndex], ordering);
            mpiSendAsync(arrayBuffer.data(), arrayBuffer.size(), destinationRank, 2 + arrayIndex, sendRequests);
            sendBuffers.emplace_back(std::move(arrayBuffer));
        }
    }

    // handle thisRank
    for (int arrayIndex = 0; arrayIndex < nArrays; ++arrayIndex)
    {
        // TODO: eliminate one copy by swapping the source with tempBuffer
        extractRange(sendList[thisRank], sourceArrays[arrayIndex], ordering, tempBuffer);
        std::copy(tempBuffer, tempBuffer + sendList[thisRank].totalCount(), destinationArrays[arrayIndex]);
    }

    unsigned nParticlesPresent = sendList[thisRank].totalCount();

    while (nParticlesPresent != nParticlesAssigned)
    {
        MPI_Status status[2 + nArrays];
        int receiveRank;
        std::size_t receiveRankCount;
        mpiRecvSync(&receiveRank, 1, MPI_ANY_SOURCE, 0, &status[0]);
        mpiRecvSync(&receiveRankCount, 1, receiveRank, 1, &status[1]);

        for (int arrayIndex = 0; arrayIndex < nArrays; ++arrayIndex)
        {
            mpiRecvSync(destinationArrays[arrayIndex] + nParticlesPresent, receiveRankCount,
                        receiveRank, 2 + arrayIndex, &status[2 + arrayIndex]);
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

/*! \brief exchange array elements with other ranks according to the specified ranges
 *
 * \tparam T                  double, float or int
 * \tparam Arrays             all std::vector<T>
 * \param sendList[in]        List of index ranges assigned to each rank, indices
 *                            are valid w.r.t to arrays present on \a thisRank relative to the \a inputOffset.
 * \param thisRank[in]        Rank of the executing process
 * \param totalSize[in]       The final size for each array
 * \param nParticlesAssigned  Number of particles for each array that participate in the exchange on \a thisRank.
 *                            This serves as the stop criterion for listening to incoming particles.
 * \param inputOffset[in]     Access arrays starting from \a inputOffset when extracting particles for sending
 * \param outputOffset[in]    Incoming particles will be added to their destination arrays starting from \a outputOffset
 * \param ordering[in]        Ordering through which to access arrays
 * \param arrays[inout]       Arrays of identical sizes, the index range based exchange operations
 *                            performed are identical for each input array. Upon completion, arrays will
 *                            contain elements from the specified ranges from all ranks.
 *                            The order in which the incoming ranges are grouped is random.
 *
 *  Example: If sendList[ri] contains the range [upper, lower), all elements (arrays+inputOffset)[upper:lower]
 *           will be sent to rank ri. At the destination ri, any assigned particles already present,
 *           are moved to their destination arrays, starting from \a outputOffset. The incoming elements to ri
 *           will be appended to the aforementioned elements.
 *           No information about incoming particles to \a thisRank is contained in the function arguments,
 *           only their total number \a nParticlesAssigned, which also includes any assigned particles
 *           already present on \a thisRank.
 */
template<class T, class... Arrays>
void exchangeParticles(const SendList& sendList, Rank thisRank, int totalSize, int nParticlesAssigned,
                       int inputOffset, int outputOffset, const int* ordering, Arrays&... arrays)
{
    std::array<std::vector<T>*, sizeof...(Arrays)> data{ (&arrays)... };
    for (auto array : data)
    {
        array->reserve(totalSize);
        array->resize(totalSize);
    }

    std::vector<T> tempBuffer(totalSize);
    exchangeParticlesImpl(sendList, thisRank, nParticlesAssigned, inputOffset, outputOffset,
                          ordering, tempBuffer.data(), arrays...);
}

/*! \brief exchange array elements with other ranks according to the specified ranges
 *
 * \tparam T                  double, float or int
 * \tparam Arrays             all std::vector<T>
 * \param sendList[in]        List of index ranges assigned to each rank, indices
 *                            are valid w.r.t to arrays present on \a thisRank
 * \param thisRank[in]        Rank of the executing process
 * \param nParticlesAssigned  Number of elements that each array will hold on \a thisRank after the exchange
 * \param ordering[in]        Ordering through which to access arrays
 * \param arrays[inout]       Arrays of identical sizes, the index range based exchange operations
 *                            performed are identical for each input array. Upon completion, arrays will
 *                            contain elements from the specified ranges from all ranks.
 *                            The order in which the incoming ranges are grouped is random.
 *
 * See documentation of exchangeParticles with the full signature
 */
template<class T, class... Arrays>
void exchangeParticles(const SendList& sendList, Rank thisRank, int nParticlesAssigned,
                       const int* ordering, Arrays&... arrays)
{
    exchangeParticles<T>(sendList, thisRank, nParticlesAssigned, nParticlesAssigned, 0, 0, ordering, arrays...);
}

} // namespace sphexa
