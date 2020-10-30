#pragma once

#include <algorithm>

#include "sfc/octree.hpp"
#include "sfc/mortoncode.hpp"

namespace sphexa
{

/*! \brief split the global tree/SFC into nSplits continuous Morton code ranges
 *
 * \tparam I                 32- or 64-bit integer
 * \param globalTree         the octree
 * \param globalCounts       counts per leaf
 * \param particlesPerSplit
 * \param nSplits
 * \return                   contiguous ranges of morton codes for each split plus the particle count
 *
 * Not the best way to distribute the global tree to different ranks, but a very simple one
 */
template<class I>
std::vector<std::tuple<I,I,int>> distributeSpaceCurve(const std::vector<I>& globalTree, const std::vector<int>& globalCounts,
                                                      int particlesPerSplit, int nSplits)
{
    std::vector<std::tuple<I,I,int>> ret(nSplits);

    int leavesDone = 0;
    for (int split = 0; split < nSplits; ++split)
    {
        int splitCount = 0;
        int j = leavesDone;
        while (splitCount < particlesPerSplit && j < nNodes(globalTree))
        {
            splitCount += globalCounts[j++];
        }
        ret[split] = std::make_tuple(globalTree[leavesDone], globalTree[j], splitCount);
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
