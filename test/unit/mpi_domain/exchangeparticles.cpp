
#include <vector>
#include <mpi.h>

#include <gtest/gtest.h>

#define USE_MPI

#include "sfc/domaindecomp.hpp"

#include "coord_samples/random.hpp"

template<class T>
void exchangeIdenticalRanges(int thisRank, int nRanks)
{
    int gridSize = 64;

    std::vector<T> x(gridSize), y(gridSize);
    std::vector<int> ordering(gridSize);

    std::iota(begin(x), end(x), 0);
    std::iota(begin(y), end(y), 0);
    std::iota(begin(ordering), end(ordering), 0);

    int segmentSize = gridSize / nRanks;

    sphexa::SendList sendList(nRanks);
    for (int rank = 0; rank < nRanks; ++rank)
    {
        int lower = rank * segmentSize;
        int upper = lower + segmentSize;

        if (rank == nRanks -1 )
            upper += gridSize % nRanks;

        sendList[rank].addRange(lower, upper);
    }

    segmentSize = sendList[thisRank].count();
    int nParticlesThisRank = segmentSize * nRanks;

    sphexa::exchangeParticles<T>(sendList, nParticlesThisRank, thisRank, ordering, x, y);

    std::vector<T> refX(nParticlesThisRank);
    for (int rank = 0; rank < nRanks; ++rank)
    {
        std::iota(begin(refX) + rank * segmentSize, begin(refX) + rank * segmentSize + segmentSize,
                  sendList[thisRank][0][0]);
    }
    std::vector<T> refY(refX);

    EXPECT_EQ(refX, x);
    EXPECT_EQ(refY, y);
}

TEST(GlobalDomain, exchangeParticles)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    exchangeIdenticalRanges<double>(rank, nRanks);
    exchangeIdenticalRanges<float>(rank, nRanks);
    exchangeIdenticalRanges<int>(rank, nRanks);
}
