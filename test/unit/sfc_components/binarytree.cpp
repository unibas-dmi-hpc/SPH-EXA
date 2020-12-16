#include <vector>

#include "gtest/gtest.h"

#include "sfc/binarytree.hpp"
#include "sfc/octree.hpp"

namespace sphexa
{

/*! Create a set of example octree leaves
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
void internal4x4x4()
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

TEST(BinaryTree, internalTreeRegular4x4x4)
{
    sphexa::internal4x4x4<unsigned>();
    sphexa::internal4x4x4<uint64_t>();
}