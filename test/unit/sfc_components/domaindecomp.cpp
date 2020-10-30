#include <algorithm>
#include <numeric>

#include "gtest/gtest.h"

#include "sfc/domaindecomp.hpp"


TEST(DomainDecomposition, distributeSFC)
{
    using CodeType = unsigned;

    int nLeaves           = 6;
    int nParticlesPerNode = 5;
    int nParticles = nLeaves * nParticlesPerNode;

    int nSplits = 2;

    std::vector<int> counts(nLeaves, nParticlesPerNode);
    counts[nLeaves - 1] = nParticlesPerNode + 1;

    std::vector<CodeType> tree(nLeaves + 1);
    std::iota(begin(tree), end(tree), 0);

    std::vector<std::tuple<CodeType, CodeType, int>> splits
        = sphexa::distributeSpaceCurve(tree, counts, nParticles/nSplits, nSplits);

    std::vector<std::tuple<CodeType, CodeType, int>> ref{{0, 3, 15}, {3, 6, 16}};

    EXPECT_EQ(ref, splits);
}

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