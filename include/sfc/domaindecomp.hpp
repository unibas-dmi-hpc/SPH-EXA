#pragma once

#include <algorithm>
#include <vector>

#include "sfc/octree.hpp"
#include "sfc/mortoncode.hpp"

namespace sphexa
{

/*! \brief Stores ranges of local particles to be sent to another rank
 *
 * \tparam I  32- or 64-bit signed or unsigned integer to store the indices
 *
 *  Used for SendRanges with index ranges referencing elements in e.g. x,y,z,h arrays.
 *  In this case, count() equals the sum of all range differences computed as rangeEnd() - rangeStart().
 *
 *  Also used for SfcRanges with index ranges referencing parts of an SFC-octree with Morton codes.
 *  In that case, count() does NOT equal sum(rangeEnd(i) - rangeStart(i), i=0...nRanges)
 */
template<class I>
class IndexRanges
{
public:
    using IndexType = I;
    using RangeType = std::array<I, 2>;

    IndexRanges() : count_(0), ranges_{} {}

    //! \brief add a local index range
    void addRange(I lower, I upper, std::size_t cnt)
    {
        assert(lower <= upper);
        ranges_.push_back({lower, upper});
        count_ += cnt;
    }

    [[nodiscard]] I rangeStart(int i) const
    {
        return ranges_[i][0];
    }

    [[nodiscard]] I rangeEnd(int i) const
    {
        return ranges_[i][1];
    }

    //! \brief the sum of number of particles in all ranges or total send count
    [[nodiscard]] const std::size_t& count() const { return count_; }

    [[nodiscard]] std::size_t nRanges() const { return ranges_.size(); }

private:

    friend bool operator==(const IndexRanges& lhs, const IndexRanges& rhs)
    {
        return lhs.count_ == rhs.count_ && lhs.ranges_ == rhs.ranges_;
    }

    std::size_t count_;
    std::vector<RangeType> ranges_;
};

template<class I>
using RankAssignment = IndexRanges<I>;

template<class I>
using SpaceCurveAssignment = std::vector<RankAssignment<I>>;

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
    // one element per rank
    SpaceCurveAssignment<I> ret(nSplits);

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

        // other distribution strategies might have more than one range per rank
        ret[split].addRange(globalTree[leavesDone], globalTree[j], splitCount);
        leavesDone = j;
    }

    return ret;
}

//! \brief stores one or multiple index ranges of local particles to send out to another rank
using SendManifest = IndexRanges<int>; // works if there are < 2^31 local particles
//! \brief SendList will contain one manifest per rank
using SendList     = std::vector<SendManifest>;

/*! \brief Based on global assignment, create the list of local particle index ranges to send to each rank
 *
 * \tparam I                 32- or 64-bit integer
 * \param assignment         global space curve assignment to ranks
 * \param mortonCodes        sorted list of morton codes for local particles present on this rank
 * \return                   for each rank, a list of index ranges into \a mortonCodes to send
 *
 * Converts the global assignment Morton code ranges into particle indices with binary search
 */
template<class I>
SendList createSendList(const SpaceCurveAssignment<I>& assignment, const std::vector<I>& mortonCodes)
{
    using IndexType = SendManifest::IndexType;
    int nRanks = assignment.size();

    SendList ret(nRanks);

    for (int rank = 0; rank < nRanks; ++rank)
    {
        SendManifest& manifest = ret[rank];
        for (int rangeIndex = 0; rangeIndex < assignment[rank].nRanges(); ++rangeIndex)
        {
            I rangeStart = assignment[rank].rangeStart(rangeIndex);
            I rangeEnd   = assignment[rank].rangeEnd(rangeIndex);

            auto lit = std::lower_bound(cbegin(mortonCodes), cend(mortonCodes), rangeStart);
            IndexType lowerParticleIndex = std::distance(cbegin(mortonCodes), lit);

            auto uit = std::lower_bound(cbegin(mortonCodes) + lowerParticleIndex, cend(mortonCodes), rangeEnd);
            IndexType upperParticleIndex = std::distance(cbegin(mortonCodes), uit);

            IndexType count = std::distance(lit, uit);
            manifest.addRange(lowerParticleIndex, upperParticleIndex, count);
        }
    }

    return ret;
}

/*! \brief create a buffer of elements to send by extracting elements from the source array
 *
 * \tparam T         float or double
 * \param manifest   contains the index ranges of \a source to put into the send buffer
 * \param source     e.g. x,y,z,h arrays
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
    for (int rangeIndex = 0; rangeIndex < manifest.nRanges(); ++ rangeIndex)
    {
        for (int i = manifest.rangeStart(rangeIndex); i < manifest.rangeEnd(rangeIndex); ++i)
        {
            sendBuffer.push_back(source[ordering[i]]);
        }
    }

    return sendBuffer;
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
