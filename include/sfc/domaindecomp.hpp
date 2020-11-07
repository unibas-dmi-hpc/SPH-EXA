#pragma once

#include <algorithm>

#include "sfc/octree.hpp"
#include "sfc/mortoncode.hpp"

#ifdef USE_MPI
#include "mpi_wrappers.hpp"
#endif

namespace sphexa
{

//! \brief References a continuous part of the global octree
template<class I>
class SfcRange
{
public:

    SfcRange() {}
    SfcRange(I start, I end, std::size_t c)
        : codeStart_(start), codeEnd_(end), count_(c)
    {
    }

    [[nodiscard]]
    I codeStart() const { return codeStart_; }

    [[nodiscard]]
    I codeEnd() const { return codeEnd_; }

    [[nodiscard]]
    std::size_t count() const { return count_; }

private:
    // non-member free function
    friend bool operator==(const SfcRange<I>& lhs, const SfcRange<I>& rhs)
    {
        return std::tie(lhs.codeStart_, lhs.codeEnd_, lhs.count_) ==
               std::tie(rhs.codeStart_, rhs.codeEnd_, rhs.count_);
    }

    //! Morton code range start
    I   codeStart_;
    //! Morton code range end
    I   codeEnd_;
    //! global count of particles in range
    std::size_t count_;
};

template<class I>
using SpaceCurveAssignment = std::vector<std::vector<SfcRange<I>>>;

/*! \brief assign the global tree/SFC to nSplits ranks, assigning to each rank only a single Morton code range
 *
 * \tparam I                 32- or 64-bit integer
 * \param globalTree         the octree
 * \param globalCounts       counts per leaf
 * \param nSplits            divide the global tree into nSplits pieces, sensible choice e.g.: nSplits == nRanks
 * \return                   a vector with nSplit elements, each element is a vector of SfcRanges of Morton codes
 *
 * This function acts on global data. All calling ranks should call this function with identical arguments.
 *
 * Not the best way to distribute the global tree to different ranks, but a very simple one
 */
template<class I>
SpaceCurveAssignment<I> singleRangeSfcSplit(const std::vector<I>& globalTree, const std::vector<std::size_t>& globalCounts,
                                            int nSplits)
{
    // one element per rank in outer vector, just one element in inner vector to store a single range per rank;
    // other distribution strategies might have more than one range per rank
    SpaceCurveAssignment<I> ret(nSplits, std::vector<SfcRange<I>>(1)); // 1 because of one range

    std::size_t globalNParticles = std::accumulate(begin(globalCounts), end(globalCounts), std::size_t(0));

    // distribute work, every rank gets global count / nSplits,
    // the remainder gets distributed one by one
    std::vector<std::size_t> nParticlesPerSplit(nSplits, globalNParticles/nSplits);
    for (int split = 0; split < globalNParticles % nSplits; ++split)
    {
        nParticlesPerSplit[split]++;
    }

    int leavesDone = 0;
    for (int split = 0; split < nSplits; ++split)
    {
        std::size_t targetCount = nParticlesPerSplit[split];
        std::size_t splitCount = 0;
        int j = leavesDone;
        while (splitCount < targetCount && j < nNodes(globalTree))
        {
            // if adding the particles of the next leaf takes us further away from
            // the target count than where we're now, we stop
            if (targetCount < splitCount + globalCounts[j] && // overshoot
                targetCount - splitCount < splitCount + globalCounts[j] - targetCount) // overshoot more than undershoot
            { break; }

            splitCount += globalCounts[j++];
        }

        if (split < nSplits - 1)
        {
            // carry over difference of particles over/under assigned to next split
            // to avoid accumulating round off
            long int delta = (long int)(targetCount) - (long int)(splitCount);
            nParticlesPerSplit[split+1] += delta;
        }
        // afaict, j < nNodes(globalTree) can only happen if there are empty nodes at the end
        else {
            for( ; j < nNodes(globalTree); ++j)
                splitCount += globalCounts[j];
        }

        ret[split][0] = SfcRange<I>(globalTree[leavesDone], globalTree[j], splitCount);
        leavesDone = j;
    }

    return ret;
}

/*! \brief Stores ranges of local particles to be sent to another rank
 *
 *  Used to build a SendList which consists of one SendManifest per rank
 */
class SendManifest
{
public:
    using RangeType = std::array<int, 2>;

    SendManifest() : count_(0), ranges_{} {}

    SendManifest(std::initializer_list<RangeType> il) : count_(0), ranges_{il}
    {
        for (const auto& range : ranges_)
        {
            count_ += range[1] - range[0];
        }
    }

    //! \brief add a local index range
    void addRange(int lower, int upper)
    {
        assert(lower <= upper);
        ranges_.push_back({lower, upper});
        count_ += upper - lower;
    }

    const RangeType& operator[](int i) const
    {
        return ranges_[i];
    }

    [[nodiscard]] auto begin() const { return std::cbegin(ranges_); }
    [[nodiscard]] auto end()   const { return std::cend(ranges_); }

    //! \brief the sum of number of particles in all ranges or total send count
    [[nodiscard]] const int& count() const { return count_; }

private:

    friend bool operator==(const SendManifest& lhs, const SendManifest& rhs)
    {
        return lhs.ranges_ == rhs.ranges_;
    }

    int count_;
    std::vector<RangeType> ranges_;
};


using SendList = std::vector<SendManifest>;

/*! \brief Based on global assignment, create the list of local particle index ranges to send to each rank
 *
 * \tparam I                 32- or 64-bit integer
 * \param assignment         global space curve assignment to ranks
 * \param mortonCodes        sorted list of morton codes for local particles present on this rank
 * \return                   for each rank, a list of index ranges into \a mortonCodes to send
 */
template<class I>
SendList createSendList(const SpaceCurveAssignment<I>& assignment, const std::vector<I>& mortonCodes)
{
    int nRanks = assignment.size();

    SendList ret(nRanks);

    for (int rank = 0; rank < nRanks; ++rank)
    {
        SendManifest manifest;
        for (int rangeIndex = 0; rangeIndex < assignment[rank].size(); ++rangeIndex)
        {
            I rangeStart = assignment[rank][rangeIndex].codeStart();
            I rangeEnd   = assignment[rank][rangeIndex].codeEnd();

            int lowerParticleIndex = std::lower_bound(cbegin(mortonCodes), cend(mortonCodes), rangeStart) -
                                        cbegin(mortonCodes);

            int upperParticleIndex = std::lower_bound(cbegin(mortonCodes) + lowerParticleIndex, cend(mortonCodes), rangeEnd) -
                                        cbegin(mortonCodes);

            manifest.addRange(lowerParticleIndex, upperParticleIndex);
        }
        ret[rank] = manifest;
    }

    return ret;
}

/*! \brief create a buffer of elements to send
 *
 * \tparam T         float or double
 * \param manifest   contains the index ranges of \a source to put into the send buffer
 * \param source     x,y,z coordinate arrays
 * \param ordering   the space curve ordering to handle unsorted source arrays
 *                   if source is space-curve-sorted, \a ordering is the trivial 0,1,...,n sequence
 * \return           the send buffer
 */
template<class T>
std::vector<T> createSendBuffer(const SendManifest& manifest, const std::vector<T>& source,
                                const std::vector<int>& ordering)
{
    int sendSize = manifest.count();

    std::vector<T> sendBuffer;
    sendBuffer.reserve(sendSize);
    for (const auto& range : manifest)
    {
        for (int i = range[0]; i < range[1]; ++i)
        {
            sendBuffer.push_back(source[ordering[i]]);
        }
    }

    return sendBuffer;
}

#ifdef USE_MPI

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
    std::vector<std::vector<T>*> data{ (&arrays)... };
    int nRanks = sendList.size();

    std::vector<std::vector<T>> sendBuffers;
    sendBuffers.reserve( data.size() * (nRanks-1));

    std::vector<MPI_Request> sendRequests;
    sendRequests.reserve( (2 + data.size()) * (nRanks-1));

    for (int destinationRank = 0; destinationRank < nRanks; ++destinationRank)
    {
        if (destinationRank == thisRank || sendList[destinationRank].count() == 0) { continue; }

        mpiSendAsync(&thisRank, 1, destinationRank, 0, sendRequests);
        mpiSendAsync(&sendList[destinationRank].count(), 1, destinationRank, 1, sendRequests);

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

    int nParticlesPresent  = sendList[thisRank].count();
    for (auto array : data)
    {
        array->reserve(nParticlesAssigned);
        array->resize(nParticlesAssigned);
    }

    while (nParticlesPresent != nParticlesAssigned)
    {
        MPI_Status status[2 + data.size()];
        int receiveRank, receiveRankCount;
        MPI_Recv(&receiveRank, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status[0]);
        MPI_Recv(&receiveRankCount, 1, MPI_INT, receiveRank, 1, MPI_COMM_WORLD, &status[1]);

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
    // on the receiver side. For this reason, a barrier is enacted here.
    // If there are no interfering messages going to be sent, it would be possible to
    // remove the barrier. But if that assumption turns out to be wrong, arising bugs
    // will be hard to detect.
    MPI_Barrier(MPI_COMM_WORLD);
}

#endif // USE_MPI

template<class I>
struct CompareX
{
    inline bool operator()(I a, I b) { return decodeMortonX(a) < decodeMortonX(b); }
};


template<class I>
void histogramX(const I* codesStart, const I* codesEnd, std::array<unsigned, 1u<<maxTreeLevel<I>{}>& histogram)
{
    constexpr int nBins = 1u<<maxTreeLevel<I>{};

    for (int bin = 0; bin < nBins; ++bin)
    {
        auto lower = std::lower_bound(codesStart, codesEnd, bin, CompareX<I>{});
        auto upper = std::upper_bound(codesStart, codesEnd, bin, CompareX<I>{});
        histogram[bin] = upper - lower;
    }
}

} // namespace sphexa
