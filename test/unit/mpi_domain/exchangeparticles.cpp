
#include <vector>
#include <mpi.h>

#include <gtest/gtest.h>

#define USE_MPI

#include "coord_samples/random.hpp"

template<class I>
void testExchangeParticles(int rank, int nRanks)
{
    int gridSize = 64;
    RegularGridCoordinates<double, I> coordinates(gridSize);


}

TEST(GlobalDomain, exchangeParticles)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    constexpr int thisExampleRanks = 2;

    //if (nRanks != thisExampleRanks)
    //    throw std::runtime_error("this test needs 2 ranks\n");

    testExchangeParticles<unsigned>(rank, nRanks);
}
