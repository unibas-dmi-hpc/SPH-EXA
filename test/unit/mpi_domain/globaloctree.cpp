
#include <vector>
#include <mpi.h>

#include <gtest/gtest.h>

#include "sfc/mortonconversions.hpp"
#include "sfc/octree_mpi.hpp"

using namespace sphexa;

template<class I>
std::vector<I> makeRegularGrid(int rank)
{
    std::vector<I> codes;

    constexpr unsigned n     = 4;
    constexpr unsigned level = 2;

    unsigned nRanks = 2;
    unsigned istart = rank * n/nRanks;
    unsigned iend   = (rank + 1) * n/nRanks;

    // a regular n x n x n grid
    for (unsigned i = istart; i < iend; ++i)
        for (unsigned j = 0; j < n; ++j)
            for (unsigned k = 0; k < n; ++k)
    {
        codes.push_back(sphexa::codeFromBox<I>(i,j,k, level));
    }

    std::sort(begin(codes), end(codes));

    return codes;
}

template<class I>
void buildTree(int rank)
{
    using sphexa::codeFromIndices;
    auto codes = makeRegularGrid<I>(rank);

    int codeRange[2] = {0, (int)codes.size()};
    int nRanges      = 1;

    int bucketSize = 8;
    auto [tree, counts] = computeOctreeGlobal(codes.data(), codes.data() + codes.size(),
                                              codeRange, nRanges, bucketSize);

    std::vector<I> refTree{
        codeFromIndices<I>({0}),
        codeFromIndices<I>({1}),
        codeFromIndices<I>({2}),
        codeFromIndices<I>({3}),
        codeFromIndices<I>({4}),
        codeFromIndices<I>({5}),
        codeFromIndices<I>({6}),
        codeFromIndices<I>({7}),
        nodeRange<I>(0)
    };

    std::vector<std::size_t> refCounts(8,8);

    EXPECT_EQ(counts, refCounts);
    EXPECT_EQ(tree, refTree);
}

TEST(GlobalTree, basicRegularTree32)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    constexpr int thisExampleRanks = 2;

    if (nRanks != thisExampleRanks)
        throw std::runtime_error("this test needs 2 ranks\n");

    buildTree<unsigned>(rank);
    buildTree<uint64_t>(rank);
}

template<class CodeType>
void computeNodeMax(int rank)
{
    // an (incomplete) tree with 4 nodes
    std::vector<CodeType> tree{0,8,16,24,32};

    // node boundaries:                 |    |     |   |           |
    std::vector<CodeType> particleCodes{ 2,4, 8,14, 20, 24,25,26,31 };

    std::vector<std::vector<float>> smoothingLs
    {
    //  |    |    |  |       |
        {1,1, 2,3, 6, 2,9,1,3}, // rank 0 smoothing lengths
        {1,2, 4,3, 5, 2,8,1,3}, // rank 1 smoothing lengths
    };

    // expected maximum per node across both ranks searching all nodes
    std::vector<float>    hMaxPerNode{2, 4, 6, 9};
    // expected outcome with restricted range
    std::vector<float>    hMaxRestricted{1, 3, 5, 8};

    int fullRange[2] = {0, (int)particleCodes.size()};

    int lowerIndex[2] = {0, 4};
    int upperIndex[2] = {4, (int)particleCodes.size()};
    int restrictedRange[2] = { lowerIndex[rank], upperIndex[rank] };

    // trivial ordering
    std::vector<int> ordering(particleCodes.size());
    std::iota(begin(ordering), end(ordering), 0);

    {
        std::vector<float> probe(hMaxPerNode.size());

        sphexa::computeNodeMaxGlobal(tree.data(), nNodes(tree), particleCodes.data(),
                                     fullRange, 1, ordering.data(),
                                     smoothingLs[rank].data(), probe.data());

        EXPECT_EQ(probe, hMaxPerNode);
    }
    {
        std::vector<float> probe(hMaxPerNode.size());

        sphexa::computeNodeMaxGlobal(tree.data(), nNodes(tree), particleCodes.data(),
                                     restrictedRange, 1, ordering.data(),
                                     smoothingLs[rank].data(), probe.data());

        EXPECT_EQ(probe, hMaxRestricted);
    }
}

TEST(GlobalTree, computeNodeMax)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    constexpr int thisExampleRanks = 2;

    if (nRanks != thisExampleRanks)
        throw std::runtime_error("this test needs 2 ranks\n");

    computeNodeMax<unsigned>(rank);
    computeNodeMax<uint64_t>(rank);
}
