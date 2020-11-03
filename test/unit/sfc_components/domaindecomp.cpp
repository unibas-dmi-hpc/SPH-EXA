#include <algorithm>
#include <numeric>

#include "gtest/gtest.h"

#include "sfc/domaindecomp.hpp"
#include "coord_samples/random.hpp"


TEST(DomainDecomposition, singleRangeSfcSplit)
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

    auto splits = sphexa::singleRangeSfcSplit(tree, counts, nParticles/nSplits, nSplits);

    sphexa::SpaceCurveAssignment<CodeType> ref{{{0, 3, 15}}, {{3, 6, 16}}};

    EXPECT_EQ(ref, splits);
}

template<class I>
void testSingleRangeSfcSplitGrid()
{
    std::vector<I> tree;

    int n = 4;
    int level = 2;
    for (unsigned i = 0; i < n; ++i)
        for (unsigned j = 0; j < n; ++j)
            for (unsigned k = 0; k < n; ++k)
            {
                tree.push_back(sphexa::detail::codeFromBox<I>({i,j,k}, level));
            }
    tree.push_back(sphexa::nodeRange<I>(0));

    std::sort(begin(tree), end(tree));

    std::vector<int> counts(sphexa::nNodes(tree), 1);

    sphexa::SpaceCurveAssignment<I> refAssignment{
        {{0, sphexa::detail::codeFromIndices<I>({4}), 32}},
        {{sphexa::detail::codeFromIndices<I>({4}), sphexa::nodeRange<I>(0), 32}}
    };

    auto assignment = sphexa::singleRangeSfcSplit(tree, counts, 32, 2);
    EXPECT_EQ(refAssignment, assignment);
}

TEST(DomainDecomposition, singleRangeSplitGrid)
{
    testSingleRangeSfcSplitGrid<unsigned>();
    testSingleRangeSfcSplitGrid<uint64_t>();
}

template<class I>
void testCreateSendListGrid()
{
    std::vector<I> codes;

    int n = 4;
    int level = 2;
    for (unsigned i = 0; i < n; ++i)
        for (unsigned j = 0; j < n; ++j)
            for (unsigned k = 0; k < n; ++k)
            {
                codes.push_back(sphexa::detail::codeFromBox<I>({i,j,k}, level));
            }

    std::sort(begin(codes), end(codes));

    sphexa::SpaceCurveAssignment<I> assignment{
        {{0, sphexa::detail::codeFromIndices<I>({4}), 32}},
        {{sphexa::detail::codeFromIndices<I>({4}), sphexa::nodeRange<I>(0), 32}}
    };

    auto sendList = sphexa::createSendList(assignment, codes);

    sphexa::SendList refSendList{ {{0, 32}}, {{32, 64}} };

    EXPECT_EQ(refSendList, sendList);
}

TEST(DomainDecomposition, createSendListGrid)
{
    testCreateSendListGrid<unsigned>();
    testCreateSendListGrid<uint64_t>();
}

template<class I>
void testSingleRangeSfcSplitRandom()
{
    int nParticles = 1003;
    int bucketSize = 64;
    RandomGaussianCoordinates<double, I> coords(nParticles, {-1,1});

    auto [tree, counts] = sphexa::computeOctree(coords.mortonCodes().data(), coords.mortonCodes().data() + nParticles,
                                                bucketSize);

    int nSplits = 4;
    auto assignment = sphexa::singleRangeSfcSplit(tree, counts, nParticles/nSplits, nSplits);

    // all splits except the last one should at least be assigned nParticles/nSplits
    for (int rank = 0; rank < nSplits -1; ++rank)
    {
        int rankCount = 0;
        for (auto& range : assignment[rank])
            rankCount += range.count();

        EXPECT_GE(rankCount, nParticles/nSplits);
    }

    auto sendList = sphexa::createSendList(assignment, coords.mortonCodes());

    int particleRecount = 0;
    for (auto& list : sendList)
        for (auto& manifest : list)
            particleRecount += manifest[1]  - manifest[0];

    // make sure that all particles present on the node got assigned to some rank
    EXPECT_EQ(nParticles, particleRecount);
}

TEST(DomainDecomposition, assignSendIntegration)
{
    testSingleRangeSfcSplitRandom<unsigned>();
    testSingleRangeSfcSplitRandom<uint64_t>();
}

template<class I>
void testCreateSendBufferGrid()
{
    std::vector<I> codes;

    int n = 4;
    int level = 2;
    for (unsigned i = 0; i < n; ++i)
        for (unsigned j = 0; j < n; ++j)
            for (unsigned k = 0; k < n; ++k)
            {
                codes.push_back(sphexa::detail::codeFromBox<I>({i,j,k}, level));
            }

    std::sort(begin(codes), end(codes));

    sphexa::SendList sendList{ {{0, 32}}, {{32, 64}} };

    std::vector<double> particles(codes.size());
    std::vector<int> ordering(codes.size());
    std::iota(begin(particles), end(particles), 0);
    std::iota(begin(ordering), end(ordering), 0);

    auto buffer0 = sphexa::createSendBuffer(sendList[0], particles, ordering);
    auto buffer1 = sphexa::createSendBuffer(sendList[1], particles, ordering);

    EXPECT_TRUE(std::equal(begin(buffer0), end(buffer0), begin(particles)));
    EXPECT_TRUE(std::equal(begin(buffer1), end(buffer1), begin(particles) + 32));
}

TEST(DomainDecomposition, createSendBufferGrid)
{
    testCreateSendBufferGrid<unsigned>();
    testCreateSendBufferGrid<uint64_t>();
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