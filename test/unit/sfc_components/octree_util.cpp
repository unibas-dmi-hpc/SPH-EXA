
#include "gtest/gtest.h"

#include "sfc/mortoncode.hpp"
#include "sfc/octree.hpp"
#include "sfc/octree_util.hpp"

using sphexa::detail::codeFromIndices;
using sphexa::detail::codeFromBox;
using sphexa::nodeRange;
using sphexa::nNodes;


//! \brief detect missing zero node
template<class CodeType>
void invariantHead()
{
    std::vector<CodeType> tree
    {
        codeFromIndices<CodeType>({1}),
        codeFromIndices<CodeType>({2}),
        codeFromIndices<CodeType>({3}),
        codeFromIndices<CodeType>({4}),
        codeFromIndices<CodeType>({5}),
        codeFromIndices<CodeType>({6}),
        codeFromIndices<CodeType>({7}),
        nodeRange<CodeType>(0)
    };

    EXPECT_FALSE(sphexa::checkOctreeInvariants(tree.data(), nNodes(tree)));
}

//! \brief detect missing end node
template<class CodeType>
void invariantTail()
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
        codeFromIndices<CodeType>({7}),
    };

    EXPECT_FALSE(sphexa::checkOctreeInvariants(tree.data(), nNodes(tree)));
}

//! \brief detect missing siblings
template<class CodeType>
void invariantSiblings()
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
            nodeRange<CodeType>(0)
        };

    EXPECT_FALSE(sphexa::checkOctreeInvariants(tree.data(), nNodes(tree)));
}

TEST(CornerstoneUtil, invariants32)
{
    invariantHead<unsigned>();
    invariantTail<unsigned>();
    invariantSiblings<unsigned>();
}

TEST(CornerstoneUtil, invariants64)
{
    invariantHead<uint64_t>();
    invariantTail<uint64_t>();
    invariantSiblings<uint64_t>();
}
