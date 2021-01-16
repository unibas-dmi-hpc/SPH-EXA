/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! \file
 * \brief octree utility tests
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * This file implements tests for OctreeMaker.
 * OctreeMaker can be used to generate octrees in cornerstone
 * format. It is only used to test the octree implementation.
 */

#include "gtest/gtest.h"

#include "cstone/mortoncode.hpp"
#include "cstone/octree.hpp"
#include "cstone/octree_util.hpp"

using namespace cstone;

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

    EXPECT_FALSE(checkOctreeInvariants(tree.data(), nNodes(tree)));
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

    EXPECT_FALSE(checkOctreeInvariants(tree.data(), nNodes(tree)));
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

    EXPECT_FALSE(checkOctreeInvariants(tree.data(), nNodes(tree)));
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

//! \brief test OctreeMaker node division
template<class I>
void octreeMakerDivide()
{
    using CodeType = I;

    // divide root node and node {7}
    auto tree = OctreeMaker<CodeType>{}.divide().divide(7).makeTree();

    std::vector<CodeType> refTree
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

    EXPECT_EQ(tree, refTree);
}

TEST(CornerstoneUtil, octreeMakerDivide32)
{
    octreeMakerDivide<unsigned>();
}

TEST(CornerstoneUtil, octreeMakerDivide64)
{
    octreeMakerDivide<uint64_t>();
}


//! \brief test OctreeMaker creation of a maximum level tree
template<class I>
void octreeMakerMaxLevel()
{
    using CodeType = I;

    std::vector<CodeType> refTree{0};
    {
        std::array<unsigned char, maxTreeLevel<uint64_t>{}> zeroIndices{0};
        for (int level = 0; level < maxTreeLevel<CodeType>{}; ++level)
        {
            auto indices = zeroIndices;
            for (int sibling = 1; sibling < 8; ++sibling)
            {
                indices[level] = sibling;
                refTree.push_back(codeFromIndices<CodeType>(indices));
            }
        }
        refTree.push_back(nodeRange<CodeType>(0));
        std::sort(begin(refTree), end(refTree));
    }

    EXPECT_TRUE(checkOctreeInvariants(refTree.data(), nNodes(refTree)));

    OctreeMaker<CodeType> octreeMaker;
    for (int level = 0; level < maxTreeLevel<I>{}; ++level)
        octreeMaker.divide({}, level);

    std::vector<CodeType> tree = octreeMaker.makeTree();
    EXPECT_EQ(tree, refTree);
}

TEST(CornerstoneUtil, octreeMakerMaxLevel32)
{
    octreeMakerMaxLevel<unsigned>();
}

TEST(CornerstoneUtil, octreeMakerMaxLevel64)
{
    octreeMakerMaxLevel<uint64_t>();
}
