
#include <vector>

#include <mpi.h>


#include <gtest/gtest.h>
#include "gtest-mpi-listener.hpp"

#define USE_MPI
#include "sfc/octree.hpp"
#include "../sfc/randombox.hpp"

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
        codes.push_back(sphexa::detail::codeFromBox<I>({i,j,k}, level));
    }

    std::sort(begin(codes), end(codes));

    return codes;
}

template<class I>
void buildTree(int rank)
{
    using sphexa::detail::codeFromIndices;
    auto codes = makeRegularGrid<I>(rank);

    int bucketSize = 8;
    auto [tree, counts] = computeOctree<I, GlobalReduce>(codes.data(), codes.data() + codes.size(), bucketSize);

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

    std::vector<int> refCounts(8,8);

    EXPECT_EQ(counts, refCounts);
    EXPECT_EQ(tree, refTree);
}

TEST(GlobalTree, basicRegularTree32)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    buildTree<unsigned>(rank);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    MPI_Init(NULL, NULL);
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    if (nRanks != 2)
        throw std::runtime_error("This example needs to be run with 2 ranks");

    // Add object that will finalize MPI on exit; Google Test owns this pointer
    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);

    // Get the event listener list.
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();

    // Remove default listener: the default printer and the default XML printer
    ::testing::TestEventListener *l = listeners.Release(listeners.default_result_printer());
    //if (rank != 0) { delete l; }

    // Adds MPI listener; Google Test owns this pointer
    listeners.Append(new GTestMPIListener::MPIWrapperPrinter(l, MPI_COMM_WORLD));

    auto ret = RUN_ALL_TESTS();

    return ret;
}
