
#include "gtest/gtest.h"

#include "coord_samples/random.hpp"
#include "sfc/domain.hpp"


template<class I, class T>
void randomGaussianDomain(int rank, int nRanks)
{

};

TEST(Domain, moreHalos)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    randomGaussianDomain<unsigned, double>(rank, nRanks);
    //randomGaussianDomain<uint64_t, double>(rank, nRanks);
    //randomGaussianDomain<unsigned, float>(rank, nRanks);
    //randomGaussianDomain<uint64_t, float>(rank, nRanks);
}
