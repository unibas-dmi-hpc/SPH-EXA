
#include "gtest/gtest.h"

#include "sfc/mortoncode.hpp"
#include "sfc/octree.hpp"

#include "randombox.hpp"

using sphexa::detail::codeFromIndices;
using sphexa::detail::codeFromBox;
using sphexa::nodeRange;
using sphexa::nNodes;

template <class I>
class ExampleOctree
{
public:
    ExampleOctree()
        : codes{
        // 000
           codeFromIndices<I>({0,1}),
           codeFromIndices<I>({0,2}),
           codeFromIndices<I>({0,3}),
           codeFromIndices<I>({0,4}),
           codeFromIndices<I>({0,5}),
           codeFromIndices<I>({0,6}),
           codeFromIndices<I>({0,7}),
        // 001
           codeFromIndices<I>({1}),
        // 010
           codeFromIndices<I>({2,1,0}),
           codeFromIndices<I>({2,1,1}),
           codeFromIndices<I>({2,1,2}),
           codeFromIndices<I>({2,1,3}),
           codeFromIndices<I>({2,1,4}),
           codeFromIndices<I>({2,1,5}),
           codeFromIndices<I>({2,1,6}),
           codeFromIndices<I>({2,1,7}),
        // 011
           codeFromIndices<I>({3}),
        // 100
           codeFromIndices<I>({4}),
        // 101
           codeFromIndices<I>({5}),
        // 110
           codeFromIndices<I>({6}),
        // 111
           codeFromIndices<I>({7, 0}),
           codeFromIndices<I>({7, 0}) + 1,
           codeFromIndices<I>({7, 1}),
           codeFromIndices<I>({7, 2}),
           codeFromIndices<I>({7, 3}),
           codeFromIndices<I>({7, 4}),
           codeFromIndices<I>({7, 5}),
           codeFromIndices<I>({7, 6, 0}),
           codeFromIndices<I>({7, 6, 1}),
           codeFromIndices<I>({7, 6, 2}),
           codeFromIndices<I>({7, 6, 3}),
           codeFromIndices<I>({7, 6, 4}),
           codeFromIndices<I>({7, 6, 5}),
           codeFromIndices<I>({7, 6, 6}),
           codeFromIndices<I>({7, 7}),
        },
        nodes{
            sphexa::SfcNode<I>{codeFromIndices<I>({0}), codeFromIndices<I>({1}), 0, 7},
            sphexa::SfcNode<I>{codeFromIndices<I>({1}), codeFromIndices<I>({1})+1, 7, 1},
            sphexa::SfcNode<I>{codeFromIndices<I>({2,1}), codeFromIndices<I>({2,2}), 8, 8},
            sphexa::SfcNode<I>{codeFromIndices<I>({3}), codeFromIndices<I>({3})+1, 16, 1},
            sphexa::SfcNode<I>{codeFromIndices<I>({4}), codeFromIndices<I>({4})+1, 17, 1},
            sphexa::SfcNode<I>{codeFromIndices<I>({5}), codeFromIndices<I>({5})+1, 18, 1},
            sphexa::SfcNode<I>{codeFromIndices<I>({6}), codeFromIndices<I>({6})+1, 19, 1},

            sphexa::SfcNode<I>{codeFromIndices<I>({7,0}), codeFromIndices<I>({7,0})+8, 20, 2},
            sphexa::SfcNode<I>{codeFromIndices<I>({7,1}), codeFromIndices<I>({7,1})+1, 22, 1},
            sphexa::SfcNode<I>{codeFromIndices<I>({7,2}), codeFromIndices<I>({7,2})+1, 23, 1},
            sphexa::SfcNode<I>{codeFromIndices<I>({7,3}), codeFromIndices<I>({7,3})+1, 24, 1},
            sphexa::SfcNode<I>{codeFromIndices<I>({7,4}), codeFromIndices<I>({7,4})+1, 25, 1},
            sphexa::SfcNode<I>{codeFromIndices<I>({7,5}), codeFromIndices<I>({7,5})+1, 26, 1},
            sphexa::SfcNode<I>{codeFromIndices<I>({7,6}), codeFromIndices<I>({7,7}),   27, 7},
            sphexa::SfcNode<I>{codeFromIndices<I>({7,7}), codeFromIndices<I>({7,7})+1, 34, 1},
        }
    {
    }

    std::vector<I> codes;
    // expected resulting tree
    std::vector<sphexa::SfcNode<I>> nodes;
};

TEST(Octree, trimExample32)
{
    using I = unsigned;
    unsigned bucketSize = 8;

    ExampleOctree<I> tree;

    auto leaves = sphexa::trimZCurve<sphexa::SfcNode>(tree.codes, bucketSize);
    EXPECT_EQ(leaves, tree.nodes);
}

TEST(Octree, trimExample64)
{
    using I = uint64_t;
    unsigned bucketSize = 8;

    ExampleOctree<I> tree;

    auto leaves = sphexa::trimZCurve<sphexa::SfcNode>(tree.codes, bucketSize);
    EXPECT_EQ(leaves, tree.nodes);
}

TEST(Octree, trim8)
{
    using I = unsigned;
    unsigned bucketSize = 8;

    std::vector<I> codes
    {
        codeFromIndices<I>({0, 1}),
        codeFromIndices<I>({0, 2}),
        codeFromIndices<I>({0, 3}),
        codeFromIndices<I>({0, 4}),
        codeFromIndices<I>({0, 5}),
        codeFromIndices<I>({0, 6}),
        codeFromIndices<I>({0, 7}),
        codeFromIndices<I>({1}),
        codeFromIndices<I>({2,1}),
    };

    {
        I code = codes[0];
        I codeLimit = codes[8];
        auto isInBox = [code](I c1_, I c2_){ return std::get<1>(sphexa::smallestCommonBox(code, c1_)) < c2_; };
        //auto ub = std::upper_bound(cbegin(codes), cbegin(codes) + 8, codeLimit, isInBox);
        //unsigned j = ub - cbegin(codes);
        EXPECT_EQ(true, isInBox(codes[6], codeLimit));
        EXPECT_EQ(false, isInBox(codes[7], codeLimit));
    }

    auto leaves = sphexa::trimZCurve<sphexa::SfcNode>(codes, bucketSize);

    sphexa::SfcNode<I> node0{codeFromIndices<I>({0}), codeFromIndices<I>({1}), 0, 7};
    sphexa::SfcNode<I> node1{codeFromIndices<I>({1}), codeFromIndices<I>({1})+1, 7, 1};
    sphexa::SfcNode<I> node2{codeFromIndices<I>({2,1}), codeFromIndices<I>({2,1})+1, 8, 1};

    EXPECT_EQ(leaves[0], node0);
    EXPECT_EQ(leaves[1], node1);
    EXPECT_EQ(leaves[2], node2);
}

TEST(Octree, trim9)
{
    using I = unsigned;
    unsigned bucketSize = 8;

    std::vector<I> codes
    {
        codeFromIndices<I>({0, 0}),
        codeFromIndices<I>({0, 1}),
        codeFromIndices<I>({0, 2}),
        codeFromIndices<I>({0, 3}),
        codeFromIndices<I>({0, 4}),
        codeFromIndices<I>({0, 5}),
        codeFromIndices<I>({0, 6}),
        codeFromIndices<I>({0, 7}),
        codeFromIndices<I>({1}),
    };

    auto leaves = sphexa::trimZCurve<sphexa::SfcNode>(codes, bucketSize);

    sphexa::SfcNode<I> node0{codeFromIndices<I>({0}), codeFromIndices<I>({1}), 0, 8};
    sphexa::SfcNode<I> node1{codeFromIndices<I>({1}), codeFromIndices<I>({1}) + 1, 8, 1};

    EXPECT_EQ(leaves[0], node0);
    EXPECT_EQ(leaves[1], node1);
}

TEST(Octree, trim10)
{
    using I = unsigned;
    unsigned bucketSize = 8;

    std::vector<I> codes
    {
        codeFromIndices<I>({0, 0}),
        codeFromIndices<I>({0, 1}),
        codeFromIndices<I>({0, 2}),
        codeFromIndices<I>({0, 3}),
        codeFromIndices<I>({0, 4}),
        codeFromIndices<I>({0, 5}),
        codeFromIndices<I>({0, 6}),
        codeFromIndices<I>({0, 7}),
        codeFromIndices<I>({1}),
        codeFromIndices<I>({1,0,0,5}),
    };

    auto leaves = sphexa::trimZCurve<sphexa::SfcNode>(codes, bucketSize);

    sphexa::SfcNode<I> node0{codeFromIndices<I>({0}), codeFromIndices<I>({1}), 0, 8};
    sphexa::SfcNode<I> node1{codeFromIndices<I>({1}), codeFromIndices<I>({1, 0, 1}), 8, 2};

    EXPECT_EQ(leaves[0], node0);
    EXPECT_EQ(leaves[1], node1);
}

TEST(Octree, trim10a)
{
    using I = unsigned;
    unsigned bucketSize = 8;

    std::vector<I> codes
    {
        codeFromIndices<I>({0, 0}),
        codeFromIndices<I>({0, 1}),
        codeFromIndices<I>({0, 2}),
        codeFromIndices<I>({0, 3}),
        codeFromIndices<I>({0, 4}),
        codeFromIndices<I>({0, 5}),
        codeFromIndices<I>({0, 6}),
        codeFromIndices<I>({0, 7}),
        codeFromIndices<I>({1,0,0,0,3}),
        codeFromIndices<I>({1,0,0,5}),
    };

    auto leaves = sphexa::trimZCurve<sphexa::SfcNode>(codes, bucketSize);

    sphexa::SfcNode<I> node0{codeFromIndices<I>({0}), codeFromIndices<I>({1}), 0, 8};
    sphexa::SfcNode<I> node1{codeFromIndices<I>({1}), codeFromIndices<I>({1, 0, 1}), 8, 2};

    EXPECT_EQ(leaves[0], node0);
    EXPECT_EQ(leaves[1], node1);
}

TEST(Octree, trim10b)
{
    using I = unsigned;
    unsigned bucketSize = 8;

    std::vector<I> codes
    {
        codeFromIndices<I>({0, 0}),
        codeFromIndices<I>({0, 1}),
        codeFromIndices<I>({0, 2}),
        codeFromIndices<I>({0, 3}),
        codeFromIndices<I>({0, 4}),
        codeFromIndices<I>({0, 5}),
        codeFromIndices<I>({0, 6}),
        codeFromIndices<I>({0, 7}),
        codeFromIndices<I>({1}),
        codeFromIndices<I>({2}),
    };

    auto leaves = sphexa::trimZCurve<sphexa::SfcNode>(codes, bucketSize);

    sphexa::SfcNode<I> node0{codeFromIndices<I>({0}), codeFromIndices<I>({1}), 0, 8};
    sphexa::SfcNode<I> node1{codeFromIndices<I>({1}), codeFromIndices<I>({1})+1, 8, 1};
    sphexa::SfcNode<I> node2{codeFromIndices<I>({2}), codeFromIndices<I>({2})+1, 9, 1};

    ASSERT_EQ(leaves.size(), 3);
    EXPECT_EQ(leaves[0], node0);
    EXPECT_EQ(leaves[1], node1);
    EXPECT_EQ(leaves[2], node2);
}

class RandomBoxTrimmer : public testing::TestWithParam<int>
{
public:
    template<class I, template <class...> class CoordinateType>
    void check(int bucketSize)
    {
        using CodeType = I;
        sphexa::Box<double> box{0, 1};

        unsigned n = 100000;

        CoordinateType<double, CodeType> randomBox(n, box);

        auto trimmedZCurve = sphexa::trimZCurve<sphexa::SfcNode>(randomBox.mortonCodes(), bucketSize);

        std::cout << "number of nodes: " << trimmedZCurve.size() << std::endl;

        // check that nodes don't overlap
        for (int ni = 0; ni + 1 < trimmedZCurve.size(); ++ni)
        {
            EXPECT_TRUE(trimmedZCurve[ni].endCode <= trimmedZCurve[ni + 1].startCode);
        }

        // check that referenced particles are within specified range
        for (const auto &node : trimmedZCurve)
        {
            for (int i = node.coordinateIndex; i < node.coordinateIndex + node.count; ++i)
            {
                // note: assumes x,y,z already normalized in [0,1]
                CodeType iCode = sphexa::morton3D<CodeType>(randomBox.x()[i], randomBox.y()[i], randomBox.z()[i]);
                EXPECT_TRUE(node.startCode <= iCode);
                EXPECT_TRUE(iCode < node.endCode);
            }
        }
    }
};

TEST_P(RandomBoxTrimmer, trimRandomUniform32)
{
    check<unsigned, RandomCoordinates>(GetParam());
}

TEST_P(RandomBoxTrimmer, trimRandomUniform64)
{
    check<uint64_t, RandomCoordinates>(GetParam());
}

TEST_P(RandomBoxTrimmer, trimRandomNormal32)
{
    check<unsigned, RandomGaussianCoordinates>(GetParam());
}

TEST_P(RandomBoxTrimmer, trimRandomNormal64)
{
    check<uint64_t, RandomGaussianCoordinates>(GetParam());
}

std::array<int, 3> bucketSizes{64, 1024, 10000};

INSTANTIATE_TEST_SUITE_P(TrimRandomBox, RandomBoxTrimmer, testing::ValuesIn(bucketSizes));


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

class RandomBoxPingPong : public testing::TestWithParam<int>
{
public:
    template<class I, template <class...> class CoordinateType>
    void check(int bucketSize)
    {
        using CodeType = I;
        sphexa::Box<double> box{0, 1};

        unsigned n = 100000;

        CoordinateType<double, CodeType> randomBox(n, box);

        std::vector<I> tree = sphexa::computeOctree(randomBox.mortonCodes().data(),
                                                    randomBox.mortonCodes().data() + n,
                                                    bucketSize);
        std::vector<int> counts(nNodes(tree));
        sphexa::computeNodeCounts(tree.data(), counts.data(), nNodes(tree),
                                  randomBox.mortonCodes().data(),
                                  randomBox.mortonCodes().data() + n);

        std::cout << "number of nodes: " << nNodes(tree) << std::endl;

        EXPECT_TRUE(sphexa::checkOctreeInvariants(tree.data(), nNodes(tree)));

        // check that referenced particles are within specified range
        for (int nodeIndex = 0; nodeIndex < nNodes(tree); ++nodeIndex)
        {
            int nodeStart = std::lower_bound(begin(randomBox.mortonCodes()), end(randomBox.mortonCodes()), tree[nodeIndex]) -
                            begin(randomBox.mortonCodes());

            if (counts[nodeIndex])
            {
                ASSERT_LT(nodeStart, n);
            }

            for (int i = nodeStart; i < counts[nodeIndex]; ++i)
            {
                // note: assumes x,y,z already normalized in [0,1]
                CodeType iCode = sphexa::morton3D<CodeType>(randomBox.x()[i], randomBox.y()[i], randomBox.z()[i]);
                EXPECT_TRUE(tree[nodeIndex] <= iCode);
                EXPECT_TRUE(iCode < tree[nodeIndex+1]);
            }
        }
    }
};

TEST_P(RandomBoxPingPong, pingPongRandomNormal32)
{
    check<unsigned, RandomGaussianCoordinates>(GetParam());
}

TEST_P(RandomBoxPingPong, pingPongRandomNormal64)
{
    check<uint64_t, RandomGaussianCoordinates>(GetParam());
}

std::array<int, 3> bucketSizesPP{64, 1024, 10000};

INSTANTIATE_TEST_SUITE_P(RandomBoxPP, RandomBoxPingPong, testing::ValuesIn(bucketSizesPP));