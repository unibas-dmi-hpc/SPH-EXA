#pragma once

#include <algorithm>

#include "sfc/mortoncode.hpp"

namespace sphexa
{

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
