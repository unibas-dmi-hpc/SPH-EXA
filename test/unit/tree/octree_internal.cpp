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

#include "cstone/tree/octree_internal.hpp"
#include "cstone/tree/octree_util.hpp"

using namespace cstone;

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

template<class I>
void checkConnectivity(const Octree<I>& fullTree)
{
    ASSERT_TRUE(fullTree.isRoot(0));

    // check all internal nodes
    for (TreeNodeIndex nodeIdx = 0; nodeIdx < fullTree.nInternalNodes(); ++nodeIdx)
    {
        ASSERT_FALSE(fullTree.isLeaf(nodeIdx));

        I prefix  = fullTree.codeStart(nodeIdx);
        int level = fullTree.level(nodeIdx);

        EXPECT_EQ(fullTree.codeEnd(nodeIdx), prefix + nodeRange<I>(level));

        for (int octant = 0; octant < 8; ++octant)
        {
            TreeNodeIndex child = fullTree.child(nodeIdx, octant);
            EXPECT_EQ(prefix + octant * nodeRange<I>(level+1), fullTree.codeStart(child));
        }

        if (!fullTree.isRoot(nodeIdx))
        {
            TreeNodeIndex parent = fullTree.parent(nodeIdx);
            EXPECT_EQ(fullTree.level(parent), level - 1);

            I parentPrefix = fullTree.codeStart(parent);
            EXPECT_EQ(parentPrefix, enclosingBoxCode(prefix, level - 1));
        }
        else
        {
            EXPECT_EQ(fullTree.codeStart(nodeIdx), 0);
            EXPECT_EQ(fullTree.level(nodeIdx), 0);
        }
    }

    // check all leaf nodes
    for (TreeNodeIndex nodeIdx = fullTree.nInternalNodes(); nodeIdx < fullTree.nTreeNodes(); ++nodeIdx)
    {
        ASSERT_TRUE(fullTree.isLeaf(nodeIdx));

        I prefix  = fullTree.codeStart(nodeIdx);
        int level = fullTree.level(nodeIdx);

        EXPECT_EQ(fullTree.codeEnd(nodeIdx), prefix + nodeRange<I>(level));

        TreeNodeIndex parent = fullTree.parent(nodeIdx);
        EXPECT_EQ(fullTree.level(parent), level - 1);

        I parentPrefix = fullTree.codeStart(parent);
        EXPECT_EQ(parentPrefix, enclosingBoxCode(prefix, level - 1));
    }
}

/*! \brief test internal octree creation from a regular 4x4x4 grid of leaves
 *
 * This creates 64 level-2 leaf nodes. The resulting internal tree should
 * have 9 nodes, the root node and the 8 level-1 nodes.
 * The children of the root point to the level-1 nodes while the children
 * of the level-1 nodes point to the leaf nodes, i.e. the tree provided for constructing,
 * which is a separate array.
 */
template<class I>
void octree4x4x4()
{
    std::vector<I> tree = makeUniformNLevelTree<I>(64, 1);

    Octree<I> fullTree;
    fullTree.update(tree.data(), tree.data() + tree.size());

    ASSERT_EQ(fullTree.nInternalNodes(), (64 - 1) / 7);
    ASSERT_EQ(fullTree.nLeaves(), 64);
    checkConnectivity(fullTree);
}

TEST(InternalOctree, octree4x4x4)
{
    octree4x4x4<unsigned>();
    octree4x4x4<uint64_t>();
}

/*! \brief test internal octree creation with an irregular leaf tree
 *
 * The leaf tree is the result of subdividing the root node, then further
 * subdividing octant 0. This results in 15 leaves, so the internal tree
 * should have two nodes: the root and the one internal level-1 node for the
 * first octant. The root points to the one internal node and to leaves 8-15.
 * The internal level-1 nodes points to leaves 0-7.
 */
template<class I>
void octreeIrregularL2()
{
    std::vector<I> tree = OctreeMaker<I>{}.divide().divide(0).makeTree();

    Octree<I> fullTree;
    fullTree.update(tree.data(), tree.data() + tree.size());

    ASSERT_EQ(fullTree.nInternalNodes(), (15 - 1) / 7);
    ASSERT_EQ(fullTree.nLeaves(), 15);
    checkConnectivity(fullTree);
}

TEST(InternalOctree, irregularL2)
{
    octreeIrregularL2<unsigned>();
    octreeIrregularL2<uint64_t>();
}

//! \brief This creates an irregular tree. Checks geometry relations between children and parents.
template<class I>
void octreeIrregularL3()
{
    std::vector<I> tree = OctreeMaker<I>{}.divide().divide(0).divide(0,2).divide(3).makeTree();

    Octree<I> fullTree;
    fullTree.update(tree.data(), tree.data() + tree.size());
    EXPECT_EQ(fullTree.nTreeNodes(), 33);
    EXPECT_EQ(fullTree.nLeaves(), 29);
    EXPECT_EQ(fullTree.nInternalNodes(), 4);

    checkConnectivity(fullTree);

    for (int i = 0; i < fullTree.nTreeNodes(); ++i)
    {
        printf("node %3d, prefix %10o, level %1d\n", i, fullTree.codeStart(i), fullTree.level(i));
    }
}

TEST(InternalOctree, irregularL3)
{
    octreeIrregularL3<unsigned>();
    //octreeIrregularL3<uint64_t>();
}