
#include "gtest/gtest.h"

#include "sfc/layout.hpp"

using namespace sphexa;

template<class I>
void computeLocalNodeRanges()
{
    // a tree with 4 subdivisions along each dimension, 64 nodes
    std::vector<I> tree = makeUniformNLevelTree<I>(64, 1);

    // two domains
    SpaceCurveAssignment<I> assignment(2);
    assignment.addRange(Rank(0), tree[0], tree[32], 64);
    assignment.addRange(Rank(1), tree[32], tree[64], 64);

    {
        int rank = 0;
        std::vector<int> nodeIndexRanges = computeLocalNodeRanges(tree, assignment, rank);
        std::vector<int> ref{0,32};
        EXPECT_EQ(nodeIndexRanges, ref);
    }
    {
        int rank = 1;
        std::vector<int> nodeIndexRanges = computeLocalNodeRanges(tree, assignment, rank);
        std::vector<int> ref{32,64};
        EXPECT_EQ(nodeIndexRanges, ref);
    }
}

TEST(Layout, LocalLayout)
{
    computeLocalNodeRanges<unsigned>();
    computeLocalNodeRanges<uint64_t>();
}


TEST(Layout, flattenNodeList)
{
    std::vector<std::vector<int>> grouped{{0,1,2}, {3,4,5}, {6}, {}};

    std::vector<int> flattened = flattenNodeList(grouped);

    std::vector<int> ref{0,1,2,3,4,5,6};
    EXPECT_EQ(flattened, ref);
}


TEST(Layout, computeLayoutBasic)
{
    int nNodes = 64;
    std::vector<std::size_t> nodeCounts(nNodes, 1);

    std::vector<int> localNodes{0,32};
    std::vector<int> halos{32, 34};

    ArrayLayout layout = computeLayout(localNodes, halos, nodeCounts);

    for (int i = 0; i < 33; ++i)
        EXPECT_EQ(layout.nodePosition(i), i);

    EXPECT_EQ(layout.nodePosition(34), 33);

    EXPECT_EQ(layout.nLocalRanges(), 1);
    EXPECT_EQ(layout.localRangePosition(0), 0);
    EXPECT_EQ(layout.localRangeCount(0), 32);
    EXPECT_EQ(layout.totalSize(), 34);
}


TEST(Layout, computeLayoutElaborate)
{
    int nNodes = 32;
    std::vector<std::size_t> nodeCounts(nNodes, 1);

    std::vector<int> localNodes{4,10,23,28};

    std::vector<int> halos{1, 3, 14, 15, 16, 21, 30};
    nodeCounts[1]  = 2;
    nodeCounts[3]  = 3;
    nodeCounts[4]  = 5;
    nodeCounts[16] = 6;
    nodeCounts[24] = 8;
    nodeCounts[30] = 9;

    ArrayLayout layout = computeLayout(localNodes, halos, nodeCounts);

    EXPECT_EQ(layout.nodePosition(1), 0);
    EXPECT_EQ(layout.nodePosition(3), 2);
    EXPECT_EQ(layout.nodePosition(4), 5);
    EXPECT_EQ(layout.nodePosition(5), 10);
    EXPECT_EQ(layout.nodePosition(6), 11);
    EXPECT_EQ(layout.nodePosition(7), 12);
    EXPECT_EQ(layout.nodePosition(8), 13);
    EXPECT_EQ(layout.nodePosition(9), 14);
    EXPECT_EQ(layout.nodePosition(14), 15);
    EXPECT_EQ(layout.nodePosition(15), 16);
    EXPECT_EQ(layout.nodePosition(16), 17);
    EXPECT_EQ(layout.nodePosition(21), 23);
    EXPECT_EQ(layout.nodePosition(23), 24);
    EXPECT_EQ(layout.nodePosition(24), 25);
    EXPECT_EQ(layout.nodePosition(25), 33);
    EXPECT_EQ(layout.nodePosition(26), 34);
    EXPECT_EQ(layout.nodePosition(27), 35);
    EXPECT_EQ(layout.nodePosition(30), 36);

    EXPECT_EQ(layout.nodeCount(1), 2);
    EXPECT_EQ(layout.nodeCount(3), 3);
    EXPECT_EQ(layout.nodeCount(4), 5);
    EXPECT_EQ(layout.nodeCount(5), 1);
    EXPECT_EQ(layout.nodeCount(6), 1);
    EXPECT_EQ(layout.nodeCount(7), 1);
    EXPECT_EQ(layout.nodeCount(8), 1);
    EXPECT_EQ(layout.nodeCount(9), 1);
    EXPECT_EQ(layout.nodeCount(14), 1);
    EXPECT_EQ(layout.nodeCount(15), 1);
    EXPECT_EQ(layout.nodeCount(16), 6);
    EXPECT_EQ(layout.nodeCount(21), 1);
    EXPECT_EQ(layout.nodeCount(23), 1);
    EXPECT_EQ(layout.nodeCount(24), 8);
    EXPECT_EQ(layout.nodeCount(25), 1);
    EXPECT_EQ(layout.nodeCount(26), 1);
    EXPECT_EQ(layout.nodeCount(27), 1);
    EXPECT_EQ(layout.nodeCount(30), 9);

    EXPECT_EQ(layout.nLocalRanges(), 2);
    EXPECT_EQ(layout.localRangePosition(0), 5);
    EXPECT_EQ(layout.localRangeCount(0), 10);
    EXPECT_EQ(layout.localRangePosition(1), 24);
    EXPECT_EQ(layout.localRangeCount(1), 12);

    EXPECT_EQ(layout.localCount(), 22);
    EXPECT_EQ(layout.totalSize(), 45);
}
