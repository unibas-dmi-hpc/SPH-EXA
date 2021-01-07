
#include <gtest/gtest.h>

#include "sfc/haloexchange.hpp"


using namespace sphexa;


template<class T>
void simpleTest(int thisRank)
{
    int nRanks = 2;
    std::vector<int> nodeList{0,1,10,11};
    std::vector<int>  offsets{0,1,3,6,10};
    ArrayLayout layout(std::move(nodeList), std::move(offsets));

    if (thisRank == 0)
        layout.addLocalRange(0,2);
    if (thisRank == 1)
        layout.addLocalRange(10,12);

    SendList incomingHalos(nRanks);
    SendList outgoingHalos(nRanks);

    if (thisRank == 0)
    {
        incomingHalos[1].addRange(3,6);
        incomingHalos[1].addRange(6,10);
        outgoingHalos[1].addRange(0,1);
        outgoingHalos[1].addRange(1,3);
    }
    if (thisRank == 1)
    {
        incomingHalos[0].addRange(0,1);
        incomingHalos[0].addRange(1,3);
        outgoingHalos[0].addRange(3,6);
        outgoingHalos[0].addRange(6,10);
    }

    EXPECT_EQ(layout.totalSize(), 10);
    std::vector<T> x(layout.totalSize());
    std::vector<T> y(layout.totalSize());

    int xshift = 20;
    int yshift = 30;
    for (int i = 0; i < layout.localRangeCount(0); ++i)
    {
        int localOffset = layout.localRangePosition(0);
        x[localOffset + i] = localOffset + i + xshift;
        y[localOffset + i] = localOffset + i + yshift;
    }

    if (thisRank == 0)
    {
        std::vector<T> xOrig{20,21,22,0,0,0,0,0,0,0};
        std::vector<T> yOrig{30,31,32,0,0,0,0,0,0,0};
        EXPECT_EQ(xOrig, x);
        EXPECT_EQ(yOrig, y);
    }
    if (thisRank == 1)
    {
        std::vector<T> xOrig{0,0,0,23,24,25,26,27,28,29};
        std::vector<T> yOrig{0,0,0,33,34,35,36,37,38,39};
        EXPECT_EQ(xOrig, x);
        EXPECT_EQ(yOrig, y);
    }

    haloexchange<T>(incomingHalos, outgoingHalos, x.data(), y.data());

    std::vector<T> xRef{20,21,22,23,24,25,26,27,28,29};
    std::vector<T> yRef{30,31,32,33,34,35,36,37,38,39};
    EXPECT_EQ(xRef, x);
    EXPECT_EQ(yRef, y);
}

TEST(HaloExchange, simpleTest)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    constexpr int thisExampleRanks = 2;

    if (nRanks != thisExampleRanks)
        throw std::runtime_error("this test needs 2 ranks\n");

    simpleTest<double>(rank);
}