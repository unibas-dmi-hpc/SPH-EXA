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

#include "cstone/octree_internal.hpp"

using namespace cstone;

template<class I>
void checkChildren(const std::vector<OctreeNode<I>>& iTree, TreeNodeIndex nodeIndex,
                   const std::vector<TreeNodeIndex>& leafParents)
{

}

/*! \brief test internal octree creation from a regular 4x4x4 grid of leaves
 *
 * This creates 64 level-2 leaf nodes. The resulting internal tree should
 * have 9 nodes, the root node and the 8 level-1 nodes.
 * The children of the root point to the level-1 nodes while the children
 * of the level-1 nodes point int the leaf nodes, which is a separate array.
 */
TEST(InternalOctree, octree4x4x4)
{
    using I = unsigned;

    std::vector<I> tree = makeUniformNLevelTree<I>(64, 1);

    InternalOctree<I> iTreeAndLeafParents = createInternalOctree(tree);
    std::vector<OctreeNode<I>>& iTree     = iTreeAndLeafParents.nodes;

    auto i = OctreeNode<I>::internal;
    auto l = OctreeNode<I>::leaf;

    std::vector<OctreeNode<I>> referenceNodes {
        {0, 0, 0, {1,2,3,4,5,6,7,8}, {i, i, i, i, i, i, i, i}}, // the root node
    };
    for (int k = 0; k < 8; ++k)
    {
        referenceNodes.push_back(
            {k * nodeRange<I>(1), 1, 0, {k*8, k*8+1, k*8+2, k*8+3, k*8+4, k*8+5, k*8+6, k*8+7}, {l, l, l, l, l, l, l, l}}
        );
    }

    // an octree with N leaves has (N-1) / 7 internal nodes
    EXPECT_EQ(iTree.size(), (nNodes(tree) - 1) / 7);

    EXPECT_EQ(iTree, referenceNodes);

    std::vector<TreeNodeIndex> refLeafParents{
        1,1,1,1,1,1,1,1,
        2,2,2,2,2,2,2,2,
        3,3,3,3,3,3,3,3,
        4,4,4,4,4,4,4,4,
        5,5,5,5,5,5,5,5,
        6,6,6,6,6,6,6,6,
        7,7,7,7,7,7,7,7,
        8,8,8,8,8,8,8,8,
    };

    std::vector<TreeNodeIndex>& leafParents = iTreeAndLeafParents.leafParents;
    EXPECT_EQ(leafParents, refLeafParents);
}


//! \brief test OctreNode equality comparison
TEST(InternalOctree, OctreeNodeEq)
{
    using I = unsigned;

    auto i = OctreeNode<I>::internal;
    auto l = OctreeNode<I>::leaf;

    OctreeNode<I> node1{0, 0, 0, {1,2,3,4,5,6,7,8}, {i, i, i, i, i, i, i, i}};
    OctreeNode<I> node2{0, 0, 0, {1,2,3,4,5,6,7,8}, {i, i, i, i, i, i, i, i}};

    EXPECT_EQ(node1, node2);

    node2 = OctreeNode<I>{1, 0, 0, {1,2,3,4,5,6,7,8}, {i, i, i, i, i, i, i, i}};
    EXPECT_FALSE(node1 == node2);

    node2 = OctreeNode<I>{0, 1, 0, {1,2,3,4,5,6,7,8}, {i, i, i, i, i, i, i, i}};
    EXPECT_FALSE(node1 == node2);

    node2 = OctreeNode<I>{0, 1, 0, {1,2,3,4,5,6,7,8}, {i, i, i, i, i, i, i, i}};
    EXPECT_FALSE(node1 == node2);

    node2 = OctreeNode<I>{0, 0, 1, {1,2,3,4,5,6,7,8}, {i, i, i, i, i, i, i, i}};
    EXPECT_FALSE(node1 == node2);

    node2 = OctreeNode<I>{0, 0, 0, {0,2,3,4,5,6,7,8}, {i, i, i, i, i, i, i, i}};
    EXPECT_FALSE(node1 == node2);

    node2 = OctreeNode<I>{0, 0, 0, {1,2,3,4,5,6,7,9}, {i, i, i, i, i, i, i, i}};
    EXPECT_FALSE(node1 == node2);

    node2 = OctreeNode<I>{0, 0, 0, {1,2,3,4,5,6,7,8}, {l, i, i, i, i, i, i, i}};
    EXPECT_FALSE(node1 == node2);

    node2 = OctreeNode<I>{0, 0, 0, {1,2,3,4,5,6,7,8}, {i, i, i, i, i, i, i, l}};
    EXPECT_FALSE(node1 == node2);
}