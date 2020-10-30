
#include "gtest/gtest.h"

#include "sfc/mortoncode.hpp"
#include "sfc/octree.hpp"

#include "coord_samples/random.hpp"

using sphexa::detail::codeFromIndices;
using sphexa::detail::codeFromBox;
using sphexa::nodeRange;
using sphexa::nNodes;

template<class I>
void printIndices(std::array<unsigned char, sphexa::maxTreeLevel<I>{}> indices)
{
    for (int i = 0; i < sphexa::maxTreeLevel<I>{}; ++i)
        std::cout << indices[i] << ",";
}


template<class T, std::size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<T, N>& input)
{
    unsigned lastNonZero = 0;
    for (int i = 0; i < N-1; ++i)
    {
        if (input[i] != 0)
        {
            lastNonZero = i;
        }
    }

    for (int i = 0; i < lastNonZero+1; ++i)
        os << int(input[i]) << ",";
    os << input[N-1];

    return os;
}

template<class I>
void printTree(const I* tree, const int* counts, int nNodes)
{
    for (int i = 0; i < nNodes; ++i)
    {
        I thisNode     = tree[i];
        //I range        = tree[i+1] - thisNode;
        //unsigned level = sphexa::treeLevel(range);
        std::cout << sphexa::detail::indicesFromCode(thisNode) << " :" << counts[i] << std::endl;
    }
    std::cout << std::endl;
}

template<class CodeType>
void checkCountTreeNodes()
{
    std::vector<CodeType> codes;

    constexpr unsigned n     = 4;
    constexpr unsigned level = 2;

    // a regular n x n x n grid
    for (unsigned i = 0; i < n; ++i)
        for (unsigned j = 0; j < n; ++j)
            for (unsigned k = 0; k < n; ++k)
    {
        codes.push_back(codeFromBox<CodeType>({i,j,k}, level));
    }

    std::sort(begin(codes), end(codes));

    std::vector<CodeType> tree{
        codeFromIndices<CodeType>({0,0}),
        codeFromIndices<CodeType>({0,1}),
        codeFromIndices<CodeType>({0,2}),
        codeFromIndices<CodeType>({0,3}),
        codeFromIndices<CodeType>({0,4}),
        codeFromIndices<CodeType>({0,5}),
        codeFromIndices<CodeType>({0,6}),
        codeFromIndices<CodeType>({0,7}),

        codeFromIndices<CodeType>({1}),
        codeFromIndices<CodeType>({2}),
        codeFromIndices<CodeType>({3}),
        codeFromIndices<CodeType>({4}),
        codeFromIndices<CodeType>({5}),
        codeFromIndices<CodeType>({6}),
        codeFromIndices<CodeType>({7}),
        nodeRange<CodeType>(0)
    };

    std::vector<int> counts(sphexa::nNodes(tree));

    sphexa::computeNodeCounts(tree.data(), counts.data(), sphexa::nNodes(tree),
                              codes.data(), codes.data() + codes.size());

    // the level 2 nodes have 1/64 of the total volume/particle count
    for (int i = 0; i < 8; ++i)
        EXPECT_EQ(counts[i], 1);

    // the level 1 nodes have 1/8 of the total
    for (int i = 8; i < counts.size(); ++i)
        EXPECT_EQ(counts[i], 8);
}

TEST(GlobalTree, countTreeNodes32)
{
    checkCountTreeNodes<unsigned>();
}

TEST(GlobalTree, countTreeNodes64)
{
    checkCountTreeNodes<uint64_t>();
}

template<class CodeType>
void rebalanceShrinkStart()
{
    constexpr int bucketSize = 8;

    std::vector<CodeType> tree;
    tree.reserve(20);

    std::vector<int> counts;
    counts.reserve(20);

    for (unsigned char i = 0; i < 8; ++i)
    {
        tree.push_back(codeFromIndices<CodeType>({0, i}));
        counts.push_back(1);
    }

    for (unsigned char i = 1; i < 8; ++i)
    {
        tree.push_back(codeFromIndices<CodeType>({i}));
        counts.push_back(1);
    }
    tree.push_back(nodeRange<CodeType>(0));

    std::vector<CodeType> balancedTree = sphexa::rebalanceTree(tree.data(), counts.data(),
                                                               sphexa::nNodes(tree), bucketSize);

    EXPECT_TRUE(sphexa::checkOctreeInvariants(balancedTree.data(), sphexa::nNodes(balancedTree)));
    EXPECT_TRUE(sphexa::checkOctreeInvariants(tree.data(), sphexa::nNodes(tree)));

    std::vector<CodeType> reference;
    reference.reserve(9);

    for (unsigned char i = 0; i < 8; ++i)
    {
        reference.push_back(codeFromIndices<CodeType>({i}));
    }
    reference.push_back(nodeRange<CodeType>(0));
    EXPECT_EQ(balancedTree, reference);
}

TEST(GlobalTree, rebalanceShrinkStart32)
{
    rebalanceShrinkStart<unsigned>();
}

TEST(GlobalTree, rebalanceShrinkStart64)
{
    rebalanceShrinkStart<uint64_t>();
}

template<class CodeType>
void rebalanceShrinkMid()
{
    constexpr int bucketSize = 8;

    std::vector<CodeType> tree
    {
        codeFromIndices<CodeType>({0}),
        codeFromIndices<CodeType>({1,0}),
        codeFromIndices<CodeType>({1,1}),
        codeFromIndices<CodeType>({1,2}),
        codeFromIndices<CodeType>({1,3}),
        codeFromIndices<CodeType>({1,4}),
        codeFromIndices<CodeType>({1,5}),
        codeFromIndices<CodeType>({1,6}),
        codeFromIndices<CodeType>({1,7}),
        codeFromIndices<CodeType>({2}),
        codeFromIndices<CodeType>({3}),
        codeFromIndices<CodeType>({4}),
        codeFromIndices<CodeType>({5}),
        codeFromIndices<CodeType>({6}),
        codeFromIndices<CodeType>({7}),
    };
    tree.push_back(nodeRange<CodeType>(0));

    std::vector<int> counts(nNodes(tree), 1);
    std::vector<CodeType> balancedTree
        = sphexa::rebalanceTree(tree.data(), counts.data(), nNodes(tree), bucketSize);

    EXPECT_TRUE(sphexa::checkOctreeInvariants(balancedTree.data(), nNodes(balancedTree)));
    EXPECT_TRUE(sphexa::checkOctreeInvariants(tree.data(), nNodes(tree)));

    std::vector<CodeType> reference;
    reference.reserve(9);

    for (unsigned char i = 0; i < 8; ++i)
    {
        reference.push_back(codeFromIndices<CodeType>({i}));
    }
    reference.push_back(nodeRange<CodeType>(0));

    EXPECT_EQ(balancedTree, reference);
}

TEST(GlobalTree, rebalanceShrinkMid32)
{
    rebalanceShrinkMid<unsigned>();
}

TEST(GlobalTree, rebalanceShrinkMid64)
{
    rebalanceShrinkMid<uint64_t>();
}

template<class CodeType>
void rebalanceShrinkEnd()
{
    constexpr int bucketSize = 8;

    std::vector<CodeType> tree
    {
        codeFromIndices<CodeType>({0}),
        codeFromIndices<CodeType>({1}),
        codeFromIndices<CodeType>({2}),
        codeFromIndices<CodeType>({3}),
        codeFromIndices<CodeType>({4}),
        codeFromIndices<CodeType>({5}),
        codeFromIndices<CodeType>({6}),
        codeFromIndices<CodeType>({7,0}),
        codeFromIndices<CodeType>({7,1}),
        codeFromIndices<CodeType>({7,2}),
        codeFromIndices<CodeType>({7,3}),
        codeFromIndices<CodeType>({7,4}),
        codeFromIndices<CodeType>({7,5}),
        codeFromIndices<CodeType>({7,6}),
        codeFromIndices<CodeType>({7,7}),
        nodeRange<CodeType>(0)
    };

    std::vector<int> counts(nNodes(tree), 1);

    std::vector<CodeType> balancedTree
        = sphexa::rebalanceTree(tree.data(), counts.data(), nNodes(tree), bucketSize);

    EXPECT_TRUE(sphexa::checkOctreeInvariants(balancedTree.data(), nNodes(balancedTree)));
    EXPECT_TRUE(sphexa::checkOctreeInvariants(tree.data(), nNodes(tree)));

    std::vector<CodeType> reference;
    reference.reserve(9);

    for (unsigned char i = 0; i < 8; ++i)
    {
        reference.push_back(codeFromIndices<CodeType>({i}));
    }
    reference.push_back(nodeRange<CodeType>(0));

    EXPECT_EQ(balancedTree, reference);
}

TEST(GlobalTree, rebalanceShrinkEnd32)
{
    rebalanceShrinkEnd<unsigned>();
}

TEST(GlobalTree, rebalanceShrinkEnd64)
{
    rebalanceShrinkEnd<uint64_t>();
}

TEST(GlobalTree, rebalanceRoot32)
{
    using CodeType = unsigned;
    constexpr int bucketSize = 8;

    // single root node
    std::vector<CodeType> tree{0, nodeRange<CodeType>(0)};
    std::vector<int>      counts{7};

    std::vector<CodeType> balancedTree
        = sphexa::rebalanceTree(tree.data(), counts.data(), nNodes(tree), bucketSize);

    EXPECT_EQ(balancedTree, tree);
}

TEST(GlobalTree, rebalanceRoot64)
{
    using CodeType = uint64_t;
    constexpr int bucketSize = 8;

    // single root node
    std::vector<CodeType> tree{0, nodeRange<CodeType>(0)};
    std::vector<int>      counts{7};

    std::vector<CodeType> balancedTree
        = sphexa::rebalanceTree(tree.data(), counts.data(), nNodes(tree), bucketSize);

    EXPECT_EQ(balancedTree, tree);
}

TEST(GlobalTree, rebalanceRootSplit32)
{
    using CodeType = unsigned;
    constexpr int bucketSize = 8;

    // single root node
    std::vector<CodeType> tree{0, nodeRange<CodeType>(0)};
    std::vector<int>      counts{9};

    std::vector<CodeType> balancedTree
        = sphexa::rebalanceTree(tree.data(), counts.data(), nNodes(tree), bucketSize);

    std::vector<CodeType> reference;
    reference.reserve(9);

    for (unsigned char i = 0; i < 8; ++i)
    {
        reference.push_back(codeFromIndices<CodeType>({i}));
    }
    reference.push_back(nodeRange<CodeType>(0));

    EXPECT_EQ(balancedTree, reference);
}

TEST(GlobalTree, rebalanceRootSplit64)
{
    using CodeType = uint64_t;
    constexpr int bucketSize = 8;

    // single root node
    std::vector<CodeType> tree{0, nodeRange<CodeType>(0)};
    std::vector<int>      counts{9};

    std::vector<CodeType> balancedTree
        = sphexa::rebalanceTree(tree.data(), counts.data(), nNodes(tree), bucketSize);

    std::vector<CodeType> reference;
    reference.reserve(9);

    for (unsigned char i = 0; i < 8; ++i)
    {
        reference.push_back(codeFromIndices<CodeType>({i}));
    }
    reference.push_back(nodeRange<CodeType>(0));

    EXPECT_EQ(balancedTree, reference);
}

template<class CodeType>
void rebalanceSplitShrink()
{
    constexpr int bucketSize = 8;

    std::vector<CodeType> tree
    {
        codeFromIndices<CodeType>({0}),
        codeFromIndices<CodeType>({1}),
        codeFromIndices<CodeType>({2}),
        codeFromIndices<CodeType>({3}),
        codeFromIndices<CodeType>({4}),
        codeFromIndices<CodeType>({5}),
        codeFromIndices<CodeType>({6}),
        codeFromIndices<CodeType>({7,0}),
        codeFromIndices<CodeType>({7,1}),
        codeFromIndices<CodeType>({7,2}),
        codeFromIndices<CodeType>({7,3}),
        codeFromIndices<CodeType>({7,4}),
        codeFromIndices<CodeType>({7,5}),
        codeFromIndices<CodeType>({7,6}),
        codeFromIndices<CodeType>({7,7}),
        nodeRange<CodeType>(0)
    };

    std::vector<int> counts(nNodes(tree), 1);
    counts[1] = bucketSize+1;

    std::vector<CodeType> balancedTree
        = sphexa::rebalanceTree(tree.data(), counts.data(), nNodes(tree), bucketSize);

    EXPECT_TRUE(sphexa::checkOctreeInvariants(balancedTree.data(), nNodes(balancedTree)));
    EXPECT_TRUE(sphexa::checkOctreeInvariants(tree.data(), nNodes(tree)));

    std::vector<CodeType> reference;
    reference.reserve(9);

    reference.push_back(codeFromIndices<CodeType>({0}));
    for (unsigned char i = 0; i < 8; ++i)
        reference.push_back(codeFromIndices<CodeType>({1,i}));
    for (unsigned char i = 2; i < 8; ++i)
        reference.push_back(codeFromIndices<CodeType>({i}));

    reference.push_back(nodeRange<CodeType>(0));

    EXPECT_EQ(balancedTree, reference);
}

TEST(GlobalTree, rebalanceSplitShrink32)
{
    rebalanceShrinkEnd<unsigned>();
}

TEST(GlobalTree, rebalanceSplitShrink64)
{
    rebalanceShrinkEnd<uint64_t>();
}

template<class CodeType>
void octreeInvariantHead()
{
    std::vector<CodeType> tree
    {
        codeFromIndices<CodeType>({1}),
        codeFromIndices<CodeType>({2}),
        codeFromIndices<CodeType>({3}),
        codeFromIndices<CodeType>({4}),
        codeFromIndices<CodeType>({5}),
        codeFromIndices<CodeType>({6}),
        codeFromIndices<CodeType>({7,0}),
        codeFromIndices<CodeType>({7,1}),
        codeFromIndices<CodeType>({7,2}),
        codeFromIndices<CodeType>({7,3}),
        codeFromIndices<CodeType>({7,4}),
        codeFromIndices<CodeType>({7,5}),
        codeFromIndices<CodeType>({7,6}),
        codeFromIndices<CodeType>({7,7}),
        nodeRange<CodeType>(0)
    };

    EXPECT_FALSE(sphexa::checkOctreeInvariants(tree.data(), nNodes(tree)));
}

template<class CodeType>
void octreeInvariantTail()
{
    std::vector<CodeType> tree
    {
        codeFromIndices<CodeType>({0}),
        codeFromIndices<CodeType>({1}),
        codeFromIndices<CodeType>({2}),
        codeFromIndices<CodeType>({3}),
        codeFromIndices<CodeType>({4}),
        codeFromIndices<CodeType>({5}),
        codeFromIndices<CodeType>({6}),
        codeFromIndices<CodeType>({7,0}),
        codeFromIndices<CodeType>({7,1}),
        codeFromIndices<CodeType>({7,2}),
        codeFromIndices<CodeType>({7,3}),
        codeFromIndices<CodeType>({7,4}),
        codeFromIndices<CodeType>({7,5}),
        codeFromIndices<CodeType>({7,6}),
        codeFromIndices<CodeType>({7,7}),
    };

    EXPECT_FALSE(sphexa::checkOctreeInvariants(tree.data(), nNodes(tree)));
}

TEST(GlobalTree, octreeInvariants32)
{
    octreeInvariantHead<unsigned>();
    octreeInvariantTail<unsigned>();
}

TEST(GlobalTree, octreeInvariants64)
{
    octreeInvariantHead<uint64_t>();
    octreeInvariantTail<uint64_t>();
}

template<class I>
void checkOctreeWithCounts(const std::vector<I>& tree, const std::vector<int>& counts, int bucketSize,
                           const std::vector<I>& mortonCodes)
{
    using CodeType = I;
    EXPECT_TRUE(sphexa::checkOctreeInvariants(tree.data(), nNodes(tree)));

    int nParticles = mortonCodes.size();

    // check that referenced particles are within specified range
    for (int nodeIndex = 0; nodeIndex < nNodes(tree); ++nodeIndex)
    {
        int nodeStart = std::lower_bound(begin(mortonCodes), end(mortonCodes), tree[nodeIndex]) -
                        begin(mortonCodes);

        int nodeEnd = std::lower_bound(begin(mortonCodes), end(mortonCodes), tree[nodeIndex+1]) -
                      begin(mortonCodes);

        // check that counts are correct
        EXPECT_EQ(nodeEnd - nodeStart, counts[nodeIndex]);
        EXPECT_LE(counts[nodeIndex], bucketSize);

        if (counts[nodeIndex])
        {
            ASSERT_LT(nodeStart, nParticles);
        }

        for (int i = nodeStart; i < counts[nodeIndex]; ++i)
        {
            CodeType iCode = mortonCodes[i];
            EXPECT_TRUE(tree[nodeIndex] <= iCode);
            EXPECT_TRUE(iCode < tree[nodeIndex+1]);
        }
    }
}

class ComputeOctreeTester : public testing::TestWithParam<int>
{
public:
    template<class I, template <class...> class CoordinateType>
    void check(int bucketSize)
    {
        using CodeType = I;
        sphexa::Box<double> box{0, 1};

        int nParticles = 100000;

        CoordinateType<double, CodeType> randomBox(nParticles, box);

        // compute octree starting from default uniform octree
        auto [treeML, countsML] = sphexa::computeOctree(randomBox.mortonCodes().data(),
                                                        randomBox.mortonCodes().data() + nParticles,
                                                        bucketSize);

        std::cout << "number of nodes: " << nNodes(treeML) << std::endl;

        checkOctreeWithCounts(treeML, countsML, bucketSize, randomBox.mortonCodes());

        // compute octree starting from just the root node
        auto [treeRN, countsRN] = sphexa::computeOctree(randomBox.mortonCodes().data(),
                                                        randomBox.mortonCodes().data() + nParticles,
                                                        bucketSize, sphexa::detail::makeRootNodeTree<I>());

        checkOctreeWithCounts(treeML, countsRN, bucketSize, randomBox.mortonCodes());

        EXPECT_EQ(treeML, treeRN);
    }
};

TEST_P(ComputeOctreeTester, pingPongRandomNormal32)
{
    check<unsigned, RandomGaussianCoordinates>(GetParam());
}

TEST_P(ComputeOctreeTester, pingPongRandomNormal64)
{
    check<uint64_t, RandomGaussianCoordinates>(GetParam());
}

std::array<int, 3> bucketSizesPP{64, 1024, 10000};

INSTANTIATE_TEST_SUITE_P(RandomBoxPP, ComputeOctreeTester, testing::ValuesIn(bucketSizesPP));