#include <vector>

#include "gtest/gtest.h"

#include "sfc/binarytree.hpp"

namespace sphexa
{

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
            0b000001ul << 56u,
            0b000010ul << 56u,
            0b000100ul << 56u,
            0b000101ul << 56u,
            0b010011ul << 56u,
            0b011000ul << 56u,
            0b011001ul << 56u,
            0b011110ul << 56u
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
void findChildrenTest()
{
    using CodeType = I;

    std::vector<CodeType> example = makeExample<CodeType>();

    std::vector<BinaryNode<CodeType>> leaves(example.size());
    for (int i = 0; i < example.size(); ++i)
    {
        leaves[i].mortonCode = example[i];
    }

    std::vector<BinaryNode<CodeType>> internalNodes(example.size() - 1);

    std::vector<BinaryNode<CodeType>*> refLeft
        {
            internalNodes.data() + 3,
            leaves.data() + 0,
            leaves.data() + 2,
            internalNodes.data() + 1,
            leaves.data() + 4,
            internalNodes.data() + 6,
            leaves.data() + 5
        };

    std::vector<BinaryNode<CodeType>*> refRight
        {
            internalNodes.data() + 4,
            leaves.data() + 1,
            leaves.data() + 3,
            internalNodes.data() + 2,
            internalNodes.data() + 5,
            leaves.data() + 7,
            leaves.data() + 6
        };

    for (int idx = 0; idx < internalNodes.size(); ++idx)
    {
        constructInternalNode(example.data(), leaves.data(), leaves.size(), internalNodes.data(), idx);
    }

    for (int idx = 0; idx < internalNodes.size(); ++idx)
    {
        EXPECT_EQ(internalNodes[idx].leftChild,  refLeft[idx]);
        EXPECT_EQ(internalNodes[idx].rightChild, refRight[idx]);
    }
}

} // namespace sphexa

TEST(BinaryTree, findChildren)
{
    sphexa::findChildrenTest<unsigned>();
    sphexa::findChildrenTest<uint64_t>();
}
