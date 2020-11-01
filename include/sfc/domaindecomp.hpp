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
