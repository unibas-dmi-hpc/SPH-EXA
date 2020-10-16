
#include "gtest/gtest.h"

#include "sfc/mortoncode.hpp"
#include "sfc/octree.hpp"

#include "randombox.hpp"

using sphexa::detail::codeFromIndices;

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

    auto leaves = sphexa::trimZCurve(tree.codes, bucketSize);
    EXPECT_EQ(leaves, tree.nodes);
}

TEST(Octree, trimExample64)
{
    using I = uint64_t;
    unsigned bucketSize = 8;

    ExampleOctree<I> tree;

    auto leaves = sphexa::trimZCurve(tree.codes, bucketSize);
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

    auto leaves = sphexa::trimZCurve(codes, bucketSize);

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

    auto leaves = sphexa::trimZCurve(codes, bucketSize);

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

    auto leaves = sphexa::trimZCurve(codes, bucketSize);

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

    auto leaves = sphexa::trimZCurve(codes, bucketSize);

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

    auto leaves = sphexa::trimZCurve(codes, bucketSize);

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

        auto trimmedZCurve = sphexa::trimZCurve(randomBox.mortonCodes(), bucketSize);

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