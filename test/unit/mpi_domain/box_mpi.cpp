
#include "gtest/gtest.h"

#include <numeric>
#include <vector>

#include "sfc/box_mpi.hpp"

using namespace sphexa;

template<class T>
void globalMin(int rank, int nRanks)
{
    int nElements = 1000;
    std::vector<T> x(nElements);
    std::iota(begin(x), end(x), 1);
    for (auto& val : x)
        val /= (rank+1);

    T gmin = globalMin(begin(x), end(x));

    EXPECT_EQ(gmin, T(1)/nRanks);
}

TEST(GlobalBox, globalMin)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    globalMin<float>(rank, nRanks);
    globalMin<double>(rank, nRanks);
}

template<class T>
void globalMax(int rank, int nRanks)
{
    int nElements = 1000;
    std::vector<T> x(nElements);
    std::iota(begin(x), end(x), 1);
    for (auto& val : x)
        val /= (rank+1);

    T gmax = globalMax(begin(x), end(x));

    EXPECT_EQ(gmax, T(nElements));
}

TEST(GlobalBox, globalMax)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    globalMax<float>(rank, nRanks);
    globalMax<double>(rank, nRanks);
}

template<class T>
void makeGlobalBox(int rank, int nRanks)
{
    int nElements = 10;
    std::vector<T> x(nElements);
    std::iota(begin(x), end(x), 1);

    std::vector<T> y = x;
    std::vector<T> z = x;

    for (auto& val : x)
        val *= (rank+1);

    for (auto& val : y)
        val *= (rank+2);

    for (auto& val : z)
        val *= (rank+3);

    Box<T> box = makeGlobalBox(begin(x), end(x), begin(y), begin(z), true, true, true);

    Box<T> refBox{1, T(nElements*nRanks), 2, T(nElements*(nRanks+1)), 3, T(nElements*(nRanks+2)), true, true, true};

    EXPECT_EQ(box, refBox);
}

TEST(GlobalBox, makeGlobalBox)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    makeGlobalBox<float>(rank, nRanks);
    makeGlobalBox<double>(rank, nRanks);
}
