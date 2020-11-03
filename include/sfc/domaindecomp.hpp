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

using SendManifest = std::vector<std::array<int, 2>>;
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
        SendManifest manifest(assignment[rank].size());
        for (int rangeIndex = 0; rangeIndex < assignment[rank].size(); ++rangeIndex)
        {
            I rangeStart = assignment[rank][rangeIndex].codeStart();
            I rangeEnd   = assignment[rank][rangeIndex].codeEnd();

            int lowerParticleIndex = std::lower_bound(cbegin(mortonCodes), cend(mortonCodes), rangeStart) -
                                        cbegin(mortonCodes);

            int upperParticleIndex = std::lower_bound(cbegin(mortonCodes) + lowerParticleIndex, cend(mortonCodes), rangeEnd) -
                                        cbegin(mortonCodes);

            manifest[rangeIndex]   = SendManifest::value_type{lowerParticleIndex, upperParticleIndex};
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
    int sendSize = 0;
    for (auto& range : manifest)
    {
        sendSize += range[1] - range[0];
    }

    std::vector<T> sendBuffer;
    sendBuffer.reserve(sendSize);
    for (auto& range : manifest)
    {
        for (int i = range[0]; i < range[1]; ++i)
        {
            sendBuffer.push_back(source[ordering[i]]);
        }
    }

    return sendBuffer;
}

template<class T, class... Arrays>
void exchangeParticles(const SendList& sendList, int receiveCount, int thisRank, Arrays&... arrays)
{
    std::vector<std::vector<T>*> data{ (&arrays)... };
}

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
