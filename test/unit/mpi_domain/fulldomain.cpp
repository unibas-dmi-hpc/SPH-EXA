
#include "gtest/gtest.h"

#include "sfc/domain.hpp"

using namespace sphexa;

template<class I, class T>
void noHalos(int rank, int nRanks)
{
    int bucketSize = 1;
    Domain<I, T> domain(rank, nRanks, bucketSize);

    std::vector<T> x{0.5, 0.6};
    std::vector<T> y{0.5, 0.6};
    std::vector<T> z{0.5, 0.6};
    std::vector<T> h{0.05, 0.05}; // no halos

    domain.sync(x,y,z,h);

    EXPECT_EQ(domain.startIndex(), 0);
    EXPECT_EQ(domain.endIndex(), 2);

    std::vector<T> cref;
    if (rank == 0)
        cref = std::vector<T>{0.5, 0.5};
    else if (rank == 1)
        cref = std::vector<T>{0.6, 0.6};

    EXPECT_EQ(cref, x);
    EXPECT_EQ(cref, y);
    EXPECT_EQ(cref, z);
}

TEST(Domain, noHalos)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    noHalos<unsigned, double>(rank, nRanks);
    noHalos<uint64_t, double>(rank, nRanks);
    noHalos<unsigned, float>(rank, nRanks);
    noHalos<uint64_t, float>(rank, nRanks);
}

template<class I, class T>
void withHalos(int rank, int nRanks)
{
    int bucketSize = 1;
    Domain<I, T> domain(rank, nRanks, bucketSize);

    std::vector<T> x{0.5, 0.6};
    std::vector<T> y{0.5, 0.6};
    std::vector<T> z{0.5, 0.6};
    std::vector<T> h{0.2, 0.22}; // in range

    domain.sync(x,y,z,h);

    if (rank == 0)
    {
        EXPECT_EQ(domain.startIndex(), 0);
        EXPECT_EQ(domain.endIndex(), 2);
    }
    else if (rank == 1)
    {
        EXPECT_EQ(domain.startIndex(), 2);
        EXPECT_EQ(domain.endIndex(), 4);
    }

    std::vector<T> cref{0.5, 0.5, 0.6, 0.6};
    std::vector<T> href{0.2, 0.2, 0.22, 0.22};

    EXPECT_EQ(cref, x);
    EXPECT_EQ(cref, y);
    EXPECT_EQ(cref, z);
    EXPECT_EQ(href, h);
}

TEST(Domain, halos)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    withHalos<unsigned, double>(rank, nRanks);
    //withHalos<uint64_t, double>(rank, nRanks);
    withHalos<unsigned, float>(rank, nRanks);
    //withHalos<uint64_t, float>(rank, nRanks);
}
