#include <vector>

#include "gtest/gtest.h"

#include "sfc/binarytree.hpp"
#include "sfc/octree.hpp"


/*! \brief add (binary) zeros behind a prefix
 *
 * Allows comparisons, such as number of leading common bits (cpr)
 * of the prefix with Morton codes.
 *
 * @tparam I      32- or 64-bit unsigned integer type
 * @param prefix  the bit pattern
 * @param length  number of bits in the prefix
 * @return        prefix padded out with zeros
 *
 */
template <class I>
constexpr I pad(I prefix, int length)
{
    return prefix << (3*sphexa::maxTreeLevel<I>{} - length);
}

TEST(BinaryTree, padUtility)
{
    EXPECT_EQ(pad(0b011,   3), 0b00011 << 27);
    EXPECT_EQ(pad(0b011ul, 3), 0b0011ul << 60);
}

namespace sphexa
{

template <class I>
void internal4x4x4PrefixTest()
{
    // a tree with 4 subdivisions along each dimension, 64 nodes
    std::vector<I> tree = detail::makeUniformNLevelTree<I>(64, 1);

    auto internalTree = createInternalTree(tree);
    EXPECT_EQ(internalTree[0].prefixLength, 0);

    EXPECT_EQ(internalTree[31].prefixLength, 1);
    EXPECT_EQ(internalTree[31].prefix, pad(I(0b0), 1));
    EXPECT_EQ(internalTree[32].prefixLength, 1);
    EXPECT_EQ(internalTree[32].prefix, pad(I(0b1), 1));

    EXPECT_EQ(internalTree[15].prefixLength, 2);
    EXPECT_EQ(internalTree[15].prefix, pad(I(0b00), 2));
    EXPECT_EQ(internalTree[16].prefixLength, 2);
    EXPECT_EQ(internalTree[16].prefix, pad(I(0b01), 2));

    EXPECT_EQ(internalTree[7].prefixLength, 3);
    EXPECT_EQ(internalTree[7].prefix, pad(I(0b000), 3));
    EXPECT_EQ(internalTree[8].prefixLength, 3);
    EXPECT_EQ(internalTree[8].prefix, pad(I(0b001), 3));
}

} // namespace sphexa

TEST(BinaryTree, internalTree4x4x4PrefixTest)
{
    sphexa::internal4x4x4PrefixTest<unsigned>();
    sphexa::internal4x4x4PrefixTest<uint64_t>();
}

namespace sphexa
{

/*! \brief Test overlap between octree nodes and coordinate ranges
 *
 * The octree node is given as a Morton code plus number of bits
 * and the coordinates as integer ranges.
 */
template <class I>
void overlapTest()
{
    // range of a level-2 node
    int r = I(1)<<(maxTreeLevel<I>{} - 2);

    BinaryNode<I> node{};

    // node range: [r,2r]^3
    node.prefix       = pad(I(0b000111), 6);
    node.prefixLength = 6;

    /// Each test is a separate case

    EXPECT_FALSE(overlap(node.prefix, node.prefixLength, Box<int>{0, r, 0, r, 0, r}));

    // exact match
    EXPECT_TRUE(overlap(node.prefix, node.prefixLength, Box<int>{r, 2*r, r, 2*r, r, 2*r}));
    // contained within (1,1,1) corner of node
    EXPECT_TRUE(overlap(node.prefix, node.prefixLength, Box<int>{2*r-1, 2*r, 2*r-1, 2*r, 2*r-1, 2*r}));
    // contained and exceeding (1,1,1) corner by 1 in all dimensions
    EXPECT_TRUE(overlap(node.prefix, node.prefixLength, Box<int>{2*r-1, 2*r+1, 2*r-1, 2*r+1, 2*r-1, 2*r+1}));

    // all of these miss the (1,1,1) corner by 1 in one of the three dimensions
    EXPECT_FALSE(overlap(node.prefix, node.prefixLength, Box<int>{2*r, 2*r+1, 2*r-1, 2*r, 2*r-1, 2*r}));
    EXPECT_FALSE(overlap(node.prefix, node.prefixLength, Box<int>{2*r-1, 2*r, 2*r, 2*r+1, 2*r-1, 2*r}));
    EXPECT_FALSE(overlap(node.prefix, node.prefixLength, Box<int>{2*r-1, 2*r, 2*r-1, 2*r, 2*r, 2*r+1}));

    // contained within (0,0,0) corner of node
    EXPECT_TRUE(overlap(node.prefix, node.prefixLength, Box<int>{r, r+1, r, r+1, r, r+1}));

    // all of these miss the (0,0,0) corner by 1 in one of the three dimensions
    EXPECT_FALSE(overlap(node.prefix, node.prefixLength, Box<int>{r-1, r, r, r+1, r, r+1}));
    EXPECT_FALSE(overlap(node.prefix, node.prefixLength, Box<int>{r, r+1, r-1, r, r, r+1}));
    EXPECT_FALSE(overlap(node.prefix, node.prefixLength, Box<int>{r, r+1, r, r+1, r-1, r}));

    // for octree leaves, we can find the number of bits in the key
    // by first computing the tree level, then multiplying by 3
    I leaf1 = pad(I(0b000111), 6);
    I leaf2 = pad(I(0b001000), 6);
    int level = treeLevel(leaf2 - leaf1);
    EXPECT_EQ(6, level * 3);
}

} // namespace sphexa

TEST(BinaryTree, internalTree4x4x4OverlapTest)
{
    sphexa::overlapTest<unsigned>();
    sphexa::overlapTest<uint64_t>();
}

namespace sphexa
{
template<class I>
void makeHaloBoxTest()
{
    {
        int r = I(1) << (maxTreeLevel<I>{} - 3);
        // node range: [r,2r]^3
        I nodeStart = pad(I(0b000000111), 9);
        I nodeEnd = pad(I(0b000001000), 9);

        /// internal node check
        {
            Box<int> haloBox = makeHaloBox(nodeStart, nodeEnd, 1, 0, 0);
            EXPECT_EQ(haloBox.xmin(), r - 1);
            EXPECT_EQ(haloBox.xmax(), 2 * r + 1);
            EXPECT_EQ(haloBox.ymin(), r);
            EXPECT_EQ(haloBox.ymax(), 2 * r);
            EXPECT_EQ(haloBox.zmin(), r);
            EXPECT_EQ(haloBox.zmax(), 2 * r);
        }
        {
            Box<int> haloBox = makeHaloBox(nodeStart, nodeEnd, 0, 1, 0);
            EXPECT_EQ(haloBox.xmin(), r);
            EXPECT_EQ(haloBox.xmax(), 2 * r);
            EXPECT_EQ(haloBox.ymin(), r - 1);
            EXPECT_EQ(haloBox.ymax(), 2 * r + 1);
            EXPECT_EQ(haloBox.zmin(), r);
            EXPECT_EQ(haloBox.zmax(), 2 * r);
        }
        {
            Box<int> haloBox = makeHaloBox(nodeStart, nodeEnd, 0, 0, 1);
            EXPECT_EQ(haloBox.xmin(), r);
            EXPECT_EQ(haloBox.xmax(), 2 * r);
            EXPECT_EQ(haloBox.ymin(), r);
            EXPECT_EQ(haloBox.ymax(), 2 * r);
            EXPECT_EQ(haloBox.zmin(), r - 1);
            EXPECT_EQ(haloBox.zmax(), 2 * r + 1);
        }
    }

    /// underflow check
    {
        int r = I(1) << (maxTreeLevel<I>{} - 1);
        // node range: [r,2r]^3
        I nodeStart = pad(I(0b000), 3);
        I nodeEnd = pad(I(0b001), 3);

        {
            Box<int> haloBox = makeHaloBox(nodeStart, nodeEnd, 1, 0, 0);
            EXPECT_EQ(haloBox.xmin(), 0);
            EXPECT_EQ(haloBox.xmax(), r+1);
        }
        {
            Box<int> haloBox = makeHaloBox(nodeStart, nodeEnd, 0, 1, 0);
            EXPECT_EQ(haloBox.ymin(), 0);
            EXPECT_EQ(haloBox.ymax(), r+1);
        }
        {
            Box<int> haloBox = makeHaloBox(nodeStart, nodeEnd, 0, 0, 1);
            EXPECT_EQ(haloBox.zmin(), 0);
            EXPECT_EQ(haloBox.zmax(), r+1);
        }
    }

    /// overflow check
    {
        int r = I(1) << (maxTreeLevel<I>{} - 1);
        // node range: [r,2r]^3
        I nodeStart = pad(I(0b111), 3);
        I nodeEnd   = nodeRange<I>(0);

        {
            Box<int> haloBox = makeHaloBox(nodeStart, nodeEnd, 1, 0, 0);
            EXPECT_EQ(haloBox.xmin(), r-1);
            EXPECT_EQ(haloBox.xmax(), 2*r);
        }
        {
            Box<int> haloBox = makeHaloBox(nodeStart, nodeEnd, 0, 1, 0);
            EXPECT_EQ(haloBox.ymin(), r-1);
            EXPECT_EQ(haloBox.ymax(), 2*r);
        }
        {
            Box<int> haloBox = makeHaloBox(nodeStart, nodeEnd, 0, 0, 1);
            EXPECT_EQ(haloBox.zmin(), r-1);
            EXPECT_EQ(haloBox.zmax(), 2*r);
        }
    }
}
} // namespace sphexa

TEST(BinaryTree, makeHaloBox)
{
    sphexa::makeHaloBoxTest<unsigned>();
    sphexa::makeHaloBoxTest<uint64_t>();
}


namespace sphexa
{
/*! \brief Traversal test for all leaves in a regular octree
 *
 * This test performs the following:
 *
 * 1. Create the leaves of a regular octree with 64 leaves and the
 *    corresponding internal binary part.
 *
 * 2. For each leaf enlarged by the halo range, find collisions
 *    between all the other leaves.
 *
 * 3. a) For each leaf, compute x,y,z coordinate ranges of the leaf + halo radius
 *    b) Test all the other leaves for overlap with the ranges of part a)
 *       If a collision between the node pair was reported in 2., there has to be overlap,
 *       if no collision was reported, there must not be any overlap.
 */
template <class I>
void regular4x4x4traversalTest()
{
    /// 1.
    // a tree with 4 subdivisions along each dimension, 64 nodes
    // node range in each dimension is 256
    std::vector<I> tree = detail::makeUniformNLevelTree<I>(64, 1);

    auto internalTree = createInternalTree(tree);

    // halo ranges
    int dx = 1;
    int dy = 1;
    int dz = 1;

    // if the box has size [0, 2^10-1]^3 (32-bit) or [0, 2^21]^3 (64-bit),
    // radius (1 + epsilon) in double will translate to radius 1 normalized to integer.
    Box<double> box(0, (1u<<maxTreeLevel<I>{})-1);
    std::vector<double> haloRadii(nNodes(tree), 1.1);

    EXPECT_EQ(dx, detail::toNBitInt<I>(normalize(haloRadii[0], box.xmin(), box.xmax())));
    EXPECT_EQ(dy, detail::toNBitInt<I>(normalize(haloRadii[0], box.ymin(), box.ymax())));
    EXPECT_EQ(dz, detail::toNBitInt<I>(normalize(haloRadii[0], box.zmin(), box.zmax())));

    /// 2.
    // find collisions of all leaf nodes enlarged by the halo ranges with all the other leaves
    // with (dx,dy,dz) = (1,1,1), this finds all immediate neighbors
    std::vector<CollisionList> collisions = findAllCollisions(internalTree, tree, haloRadii, box);

    /// 3. a)
    for (int leafIdx = 0; leafIdx < nNodes(tree); ++leafIdx)
    {
        Box<int> haloBox = makeHaloBox(tree[leafIdx], tree[leafIdx+1], dx, dy, dz);

        // number of nearest neighbors in a regular 3D grid is between 8 and 27
        EXPECT_GE(collisions[leafIdx].size(), 8);
        EXPECT_LE(collisions[leafIdx].size(), 27);

        /// 3. b)
        for (int cIdx = 0; cIdx < nNodes(tree); ++cIdx)
        {
            int collisionNodeIndex   = collisions[leafIdx][cIdx];
            I collisionLeafCode      = tree[collisionNodeIndex];
            I collisionLeafCodeUpper = tree[collisionNodeIndex+1];
            int nBits = treeLevel(collisionLeafCodeUpper - collisionLeafCode) * 3;

            // has a collision been reported between leafNodes cIdx and leafIdx?
            bool hasCollision =
                std::find(collisions[leafIdx].begin(), collisions[leafIdx].end(), collisionNodeIndex) != collisions[leafIdx].end();

            if (hasCollision)
            {
                // if yes, then the cIdx nodes has to overlap with leafIdx enlarged by the halos
                EXPECT_TRUE(overlap(collisionLeafCode, nBits, haloBox));
            }
            else
            {
                // if not, then there must not be any overlap
                EXPECT_FALSE(overlap(collisionLeafCode, nBits, haloBox));
            }
        }
    }
}
} // namespace sphexa

TEST(BinaryTree, regularTree4x4x4FullTraversal)
{
    sphexa::regular4x4x4traversalTest<unsigned>();
    sphexa::overlapTest<uint64_t>();
}

namespace sphexa
{

}

TEST(BinaryTree, irregularTreeTraversal)
{

}

namespace sphexa
{

/*! Create a set of irregular octree leaves which do not cover the whole space
 *
 * This example is illustrated in the original paper referenced in sfc/binarytree.hpp.
 * Please refer to the publication for a graphical illustration of the resulting
 * node connectivity.
 */

template<class I>
std::vector<I> makeExample();

template<>
std::vector<unsigned> makeExample()
{
    std::vector<unsigned> ret
        {
            0b0000001u << 25u,
            0b0000010u << 25u,
            0b0000100u << 25u,
            0b0000101u << 25u,
            0b0010011u << 25u,
            0b0011000u << 25u,
            0b0011001u << 25u,
            0b0011110u << 25u
        };
    return ret;
}

template<>
std::vector<uint64_t> makeExample()
{
    std::vector<uint64_t> ret
        {
            0b000001ul << 58u,
            0b000010ul << 58u,
            0b000100ul << 58u,
            0b000101ul << 58u,
            0b010011ul << 58u,
            0b011000ul << 58u,
            0b011001ul << 58u,
            0b011110ul << 58u
        };
    return ret;
}

template<class I>
void findSplitTest()
{
    std::vector<I> example = makeExample<I>();

    {
        int split = findSplit(example.data(), 0, 7);
        EXPECT_EQ(split, 3);
    }
    {
        int split = findSplit(example.data(), 0, 3);
        EXPECT_EQ(split, 1);
    }
    {
        int split = findSplit(example.data(), 4, 7);
        EXPECT_EQ(split, 4);
    }
}

TEST(BinaryTree, findSplit)
{
    findSplitTest<unsigned>();
    findSplitTest<uint64_t>();
}

template<class I>
void paperExampleTest()
{
    using CodeType = I;

    std::vector<CodeType> example = makeExample<CodeType>();
    std::vector<BinaryNode<CodeType>> internalNodes = createInternalTree(example);

    std::vector<BinaryNode<CodeType>*> refLeft
        {
            internalNodes.data() + 3,
            nullptr,
            nullptr,
            internalNodes.data() + 1,
            nullptr,
            internalNodes.data() + 6,
            nullptr
        };

    std::vector<BinaryNode<CodeType>*> refRight
        {
            internalNodes.data() + 4,
            nullptr,
            nullptr,
            internalNodes.data() + 2,
            internalNodes.data() + 5,
            nullptr,
            nullptr
        };

    std::vector<int> refLeftIndices {-1, 0, 2, -1, 4, -1, 5};
    std::vector<int> refRightIndices{-1, 1, 3, -1, -1, 7, 6};

    std::vector<int> refPrefixLengths{0, 3, 4, 2, 1, 2, 4};

    for (int idx = 0; idx < internalNodes.size(); ++idx)
    {
        EXPECT_EQ(internalNodes[idx].leftChild,      refLeft[idx]);
        EXPECT_EQ(internalNodes[idx].leftLeafIndex,  refLeftIndices[idx]);
        EXPECT_EQ(internalNodes[idx].rightChild,     refRight[idx]);
        EXPECT_EQ(internalNodes[idx].rightLeafIndex, refRightIndices[idx]);
        EXPECT_EQ(internalNodes[idx].prefixLength,   refPrefixLengths[idx]);
    }
}

} // namespace sphexa

TEST(BinaryTree, internalIrregular)
{
    sphexa::paperExampleTest<unsigned>();
    sphexa::paperExampleTest<uint64_t>();
}
