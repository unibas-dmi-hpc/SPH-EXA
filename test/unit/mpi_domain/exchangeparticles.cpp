
#include <vector>
#include <mpi.h>

#include <gtest/gtest.h>

#define USE_MPI

#include "sfc/domaindecomp.hpp"

#include "coord_samples/random.hpp"

template<class I>
void testExchangeParticles(int rank, int nRanks)
{
    int gridSize = 64;

    std::vector<double> x(gridSize);
    std::vector<int>    ordering(gridSize);

    std::iota(begin(x), end(x), 0);
    std::iota(begin(ordering), end(ordering), 0);

    sphexa::SendList sendList{ {{0, 32}}, {{32, 64}} };

    EXPECT_EQ(sendList[0].count(), 32);
    EXPECT_EQ(sendList[1].count(), 32);

    sphexa::exchangeParticles<double>(sendList, gridSize, rank, ordering, x);

    if (rank == 0)
    {
        std::vector<double> refX(gridSize);
        std::iota(begin(refX), begin(refX) + 32, 0);
        std::iota(begin(refX)+32, end(refX), 0);
        EXPECT_EQ(refX, x);
    }
    else
    {
        std::vector<double> refX(gridSize);
        std::iota(begin(refX), begin(refX) + 32, 32);
        std::iota(begin(refX)+32, end(refX), 32);
        EXPECT_EQ(refX, x);
    }
}

TEST(GlobalDomain, exchangeParticles)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    constexpr int thisExampleRanks = 2;

    if (nRanks != thisExampleRanks)
        throw std::runtime_error("this test needs 2 ranks\n");

    testExchangeParticles<unsigned>(rank, nRanks);
}
