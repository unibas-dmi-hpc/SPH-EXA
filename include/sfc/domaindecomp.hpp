#pragma once

#include <algorithm>

#include "sfc/octree.hpp"
#include "sfc/mortoncode.hpp"

namespace sphexa
{

//! \brief Represents a continuous part of the global octree
template<class I>
class SfcRange
{
public:

    SfcRange() {}
    SfcRange(I start, I end, int c)
        : codeStart_(start), codeEnd_(end), count_(c)
    {
    }

    [[nodiscard]]
    I codeStart() const { return codeStart_; }

    [[nodiscard]]
    I codeEnd() const { return codeEnd_; }

    [[nodiscard]]
    int count() const { return count_; }

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
    int count_;
};

template<class I>
using SpaceCurveAssignment = std::vector<std::vector<SfcRange<I>>>;

/*! \brief assign the global tree/SFC to nSplits ranks, assigning to each rank only a single Morton code range
 *
 * \tparam I                 32- or 64-bit integer
 * \param globalTree         the octree
 * \param globalCounts       counts per leaf
 * \param particlesPerSplit  number of particles to put in each split, sensible choice e.g.: sum(globalCounts)/nSplits
 * \param nSplits            divide the global tree into nSplits pieces, sensible choice e.g.: nSplits == nRanks
 * \return                   a vector with nSplit elements, each element is a vector of SfcRanges of Morton codes
 *
 * Not the best way to distribute the global tree to different ranks, but a very simple one
 */
template<class I>
SpaceCurveAssignment<I> singleRangeSfcSplit(const std::vector<I>& globalTree, const std::vector<int>& globalCounts,
                                            int particlesPerSplit, int nSplits)
{
    // one element per rank in outer vector, just one element in inner vector to store a single range per rank;
    // other distribution strategies might have more than one range per rank
    SpaceCurveAssignment<I> ret(nSplits, std::vector<SfcRange<I>>(1));

    int leavesDone = 0;
    for (int split = 0; split < nSplits; ++split)
    {
        int splitCount = 0;
        int j = leavesDone;
        while (splitCount < particlesPerSplit && j < nNodes(globalTree))
        {
            splitCount += globalCounts[j++];
        }
        ret[split][0] = SfcRange<I>(globalTree[leavesDone], globalTree[j], splitCount);
        leavesDone = j;
    }

    return ret;
}

class SendManifest_
{
public:
    using RangeType = std::array<int, 2>;

    SendManifest_() : count_(0), ranges_{} {}

    SendManifest_(std::initializer_list<RangeType> il) : count_(0), ranges_{il}
    {
        for (const auto& range : ranges_)
        {
            count_ += range[1] - range[0];
        }
    }

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

    //int& count() { return count_; }

    [[nodiscard]]
    const int& count() const { return count_; }

private:

    friend bool operator==(const SendManifest_& lhs, const SendManifest_& rhs)
    {
        return lhs.ranges_ == rhs.ranges_;
    }

    int count_;
    std::vector<RangeType> ranges_;
};


//using SendManifest = std::vector<std::array<int, 2>>;
using SendManifest = SendManifest_;
using SendList     = std::vector<SendManifest>;

/*! \brief create the list of particle index ranges to send to each rank
 *
 * \tparam I                 32- or 64-bit integer
 * \param assignment         space curve assignment to ranks
 * \param mortonCodes        sorted list of morton codes for particles present on this rank
 * \return                   for each rank, a list of index ranges into \a mortonCodes to send
 */
template<class I>
SendList createSendList(const SpaceCurveAssignment<I>& assignment, const std::vector<I>& mortonCodes)
{
    int nRanks = assignment.size();

    SendList ret(nRanks);

    for (int rank = 0; rank < nRanks; ++rank)
    {
        SendManifest manifest;//(assignment[rank].size());
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

template<class T>
std::enable_if_t<std::is_same<double, std::decay_t<T>>{}>
mpiSendAsync(T* data, int count, int rank, int tag, std::vector<MPI_Request>& requests)
{
    requests.push_back(MPI_Request{});
    MPI_Isend(data, count, MPI_DOUBLE, rank, tag, MPI_COMM_WORLD, &requests.back());
}

template<class T>
std::enable_if_t<std::is_same<float, std::decay_t<T>>{}>
mpiSendAsync(T* data, int count, int rank, int tag, std::vector<MPI_Request>& requests)
{
    requests.push_back(MPI_Request{});
    MPI_Isend(data, count, MPI_FLOAT, rank, tag, MPI_COMM_WORLD, &requests.back());
}

template<class T>
std::enable_if_t<std::is_same<int, std::decay_t<T>>{}>
mpiSendAsync(T* data, int count, int rank, int tag, std::vector<MPI_Request>& requests)
{
    requests.push_back(MPI_Request{});
    MPI_Isend(data, count, MPI_INT, rank, tag, MPI_COMM_WORLD, &requests.back());
}

template<class T, class... Arrays>
void exchangeParticles(const SendList& sendList, int receiveCount, int thisRank, const std::vector<int>& ordering, Arrays&... arrays)
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
        array->resize(receiveCount);
    }

    while (nParticlesPresent != receiveCount)
    {
        assert(nParticlesPresent < receiveCount);
        MPI_Status status[2 + data.size()];
        int receiveRank, receiveRankCount;
        MPI_Recv(&receiveRank, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status[0]);
        MPI_Recv(&receiveRankCount, 1, MPI_INT, receiveRank, 1, MPI_COMM_WORLD, &status[1]);
        assert(nParticlesPresent + receiveRankCount <= receiveCount);

        for (int arrayIndex = 0; arrayIndex < data.size(); ++arrayIndex)
        {
            if constexpr (std::is_same<T, double>{})
            {
                MPI_Recv(data[arrayIndex]->data() + nParticlesPresent, receiveRankCount, MPI_DOUBLE, receiveRank, 2 + arrayIndex,
                         MPI_COMM_WORLD, &status[2 + arrayIndex]);
            }
            else if constexpr (std::is_same<T, float>{})
            {
                MPI_Recv(data[arrayIndex]->data() + nParticlesPresent, receiveRankCount, MPI_FLOAT, receiveRank, 2 + arrayIndex,
                         MPI_COMM_WORLD, &status[2 + arrayIndex]);
            }
        }

        nParticlesPresent += receiveRankCount;
    }

    if (not sendRequests.empty())
    {
        MPI_Status status[sendRequests.size()];
        MPI_Waitall(sendRequests.size(), sendRequests.data(), status);
    }
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
