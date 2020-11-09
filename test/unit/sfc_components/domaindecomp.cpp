#include <algorithm>
#include <numeric>

#include "gtest/gtest.h"

#include "sfc/domaindecomp.hpp"
#include "coord_samples/random.hpp"


TEST(DomainDecomposition, singleRangeSfcSplit)
{
    using CodeType = unsigned;
    {
        int nSplits = 2;
        std::vector<std::size_t> counts{5 ,5, 5, 5, 5, 6};
        std::vector<CodeType>    tree{0, 1, 2, 3, 4, 5, 6};

        auto splits = sphexa::singleRangeSfcSplit(tree, counts, nSplits);

        sphexa::SpaceCurveAssignment<CodeType> ref(nSplits);
        ref[0].addRange(0,3,15);
        ref[1].addRange(3,6,16);
        EXPECT_EQ(ref, splits);
    }
    {
        int nSplits = 2;
        std::vector<std::size_t> counts{5, 5, 5, 15, 1, 0};
        std::vector<CodeType>    tree{0, 1, 2, 3, 4, 5, 6};

        auto splits = sphexa::singleRangeSfcSplit(tree, counts, nSplits);

        sphexa::SpaceCurveAssignment<CodeType> ref(nSplits);
        ref[0].addRange(0,3,15);
        ref[1].addRange(3,6,16);
        EXPECT_EQ(ref, splits);
    }
    {
        int nSplits = 2;
        std::vector<std::size_t> counts{15, 0, 1, 5, 5, 5};
        std::vector<CodeType>    tree{0, 1, 2, 3, 4, 5, 6};

        auto splits = sphexa::singleRangeSfcSplit(tree, counts, nSplits);

        sphexa::SpaceCurveAssignment<CodeType> ref(nSplits);
        ref[0].addRange(0,3,16);
        ref[1].addRange(3,6,15);
        EXPECT_EQ(ref, splits);
    }
    {
        int nSplits = 7;
        std::vector<std::size_t> counts{4, 3, 4, 3, 4, 3, 4, 3, 4, 3};
        // should be grouped |4|7|3|7|4|7|3|
        std::vector<CodeType> tree{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

        auto splits = sphexa::singleRangeSfcSplit(tree, counts, nSplits);

        sphexa::SpaceCurveAssignment<CodeType> ref(nSplits);
        ref[0].addRange(0,1,4);
        ref[1].addRange(1,3,7);
        ref[2].addRange(3,4,3);
        ref[3].addRange(4,6,7);
        ref[4].addRange(6,7,4);
        ref[5].addRange(7,9,7);
        ref[6].addRange(9,10,3);
        EXPECT_EQ(ref, splits);
    }
}

template<class I>
void singleRangeSfcSplitGrid()
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

    std::vector<std::size_t> counts(sphexa::nNodes(tree), 1);

    sphexa::SpaceCurveAssignment<I> refAssignment(2);
    refAssignment[0].addRange(0, sphexa::detail::codeFromIndices<I>({4}), 32);
    refAssignment[1].addRange(sphexa::detail::codeFromIndices<I>({4}), sphexa::nodeRange<I>(0), 32);

    auto assignment = sphexa::singleRangeSfcSplit(tree, counts, 2);
    EXPECT_EQ(refAssignment, assignment);
}

TEST(DomainDecomposition, singleRangeSplitGrid)
{
    singleRangeSfcSplitGrid<unsigned>();
    singleRangeSfcSplitGrid<uint64_t>();
}

template<class I>
void createSendListGrid()
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

    sphexa::SpaceCurveAssignment<I> assignment(2);
    assignment[0].addRange(0, sphexa::detail::codeFromIndices<I>({4}), 32);
    assignment[1].addRange(sphexa::detail::codeFromIndices<I>({4}), sphexa::nodeRange<I>(0), 32);

    auto sendList = sphexa::createSendList(assignment, codes);

    sphexa::SendList refSendList(2);
    refSendList[0].addRange(0,32,32);
    refSendList[1].addRange(32, 64, 32);

    EXPECT_EQ(refSendList, sendList);
    EXPECT_EQ(sendList[0].count(), 32);
    EXPECT_EQ(sendList[1].count(), 32);
}

TEST(DomainDecomposition, createSendListGrid)
{
    createSendListGrid<unsigned>();
    createSendListGrid<uint64_t>();
}

template<class I>
void singleRangeSfcSplitRandom()
{
    int nParticles = 1003;
    int bucketSize = 64;
    RandomGaussianCoordinates<double, I> coords(nParticles, {-1,1});

    auto [tree, counts] = sphexa::computeOctree(coords.mortonCodes().data(), coords.mortonCodes().data() + nParticles,
                                                bucketSize);

    int nSplits = 4;
    auto assignment = sphexa::singleRangeSfcSplit(tree, counts, nSplits);

    // all splits except the last one should at least be assigned nParticles/nSplits
    for (int rank = 0; rank < nSplits; ++rank)
    {
        std::size_t rankCount = assignment[rank].count();

        // particles in each rank should be within avg per rank +- bucketCount
        EXPECT_LE(nParticles/nSplits - bucketSize, rankCount);
        EXPECT_LE(rankCount, nParticles/nSplits + bucketSize);
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
    singleRangeSfcSplitRandom<unsigned>();
    singleRangeSfcSplitRandom<uint64_t>();
}

template<class I>
void createSendBufferGrid()
{
    std::vector<I> codes;

    // a regular grid of 64 Morton codes
    int n = 4;
    int level = 2;
    for (unsigned i = 0; i < n; ++i)
        for (unsigned j = 0; j < n; ++j)
            for (unsigned k = 0; k < n; ++k)
            {
                codes.push_back(sphexa::detail::codeFromBox<I>({i,j,k}, level));
            }

    std::sort(begin(codes), end(codes));

    sphexa::SendList sendList(2);
    sendList[0].addRange(0, 32, 32);
    sendList[1].addRange(32, 64, 32);

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
    createSendBufferGrid<unsigned>();
    createSendBufferGrid<uint64_t>();
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