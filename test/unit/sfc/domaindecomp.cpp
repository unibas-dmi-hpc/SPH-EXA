#include "gtest/gtest.h"

#include "sfc/domaindecomp.hpp"


TEST(DomainDecomposition, binX)
{
    using CodeType = unsigned;
    std::vector<CodeType> codes;

    constexpr unsigned nBins = 1u<<sphexa::maxTreeLevel<CodeType>{};

    for (unsigned i = 0; i < nBins/2; ++i)
    {
        codes.push_back(sphexa::detail::codeFromBox<CodeType>({i,0,0}, sphexa::maxTreeLevel<CodeType>{}));
        codes.push_back(sphexa::detail::codeFromBox<CodeType>({i,0,1}, sphexa::maxTreeLevel<CodeType>{}));
        codes.push_back(sphexa::detail::codeFromBox<CodeType>({i+512,0,1}, sphexa::maxTreeLevel<CodeType>{}));
        codes.push_back(sphexa::detail::codeFromBox<CodeType>({i+512,1,1}, sphexa::maxTreeLevel<CodeType>{}));
    }

    std::array<unsigned, nBins> xCounts{};

    std::sort(codes.data(), codes.data()+codes.size(), sphexa::CompareX<CodeType>{});
    sphexa::histogramX(codes.data(), codes.data()+codes.size(), xCounts);

    for (auto c : xCounts)
    {
        EXPECT_EQ(c, 2);
    }
}