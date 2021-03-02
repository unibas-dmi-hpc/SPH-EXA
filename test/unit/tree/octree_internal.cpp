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

/*! \brief test internal octree creation from a regular 4x4x4 grid of leaves
 *
 * This creates 64 level-2 leaf nodes. The resulting internal tree should
 * have 9 nodes, the root node and the 8 level-1 nodes.
 * The children of the root point to the level-1 nodes while the children
 * of the level-1 nodes point to the leaf nodes, i.e. the tree provided for constructing,
 * which is a separate array.
 */
template<class I>
void internalOctree4x4x4()
{
    std::vector<I> tree = makeUniformNLevelTree<I>(64, 1);

    std::vector<OctreeNode<I>> iTree;
    std::vector<TreeNodeIndex> leafParents;
    std::tie(iTree, leafParents) = createInternalOctree(tree);

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

    EXPECT_EQ(leafParents, refLeafParents);
}

TEST(InternalOctree, octree4x4x4)
{
    internalOctree4x4x4<unsigned>();
    internalOctree4x4x4<uint64_t>();
}

/*! \brief test internal octree creation with an irregular leaf tree
 *
 * The leaf tree is the result of subdiving the root node, then further
 * subdividing octant 0. This results in 15 leaves, so the internal tree
 * should have two nodes: the root and the one internal level-1 node for the
 * first octant. The root points to the one internal node and to leaves 8-15.
 * The internal level-1 nodes points to leaves 0-7.
 */
template<class I>
void internalOctreeIrregular()
{
    std::vector<I> tree = OctreeMaker<I>{}.divide().divide(0).makeTree();

    std::vector<OctreeNode<I>> iTree;
    std::vector<TreeNodeIndex> leafParents;
    std::tie(iTree, leafParents) = createInternalOctree(tree);

    auto i = OctreeNode<I>::internal;
    auto l = OctreeNode<I>::leaf;

    std::vector<OctreeNode<I>> referenceNodes {
            {0, 0, 0, {1,8,9,10,11,12,13,14}, {i, l, l, l, l, l, l, l}}, // the root node
            {0, 1, 0, {0,1,2,3,4,5,6,7}, {l, l, l, l, l, l, l, l}},
    };

    std::vector<TreeNodeIndex> refLeafParents{1,1,1,1,1,1,1,1,0,0,0,0,0,0,0};

    // an octree with N leaves has (N-1) / 7 internal nodes
    EXPECT_EQ(iTree.size(), (nNodes(tree) - 1) / 7);
    EXPECT_EQ(iTree, referenceNodes);
    EXPECT_EQ(leafParents, refLeafParents);
}

TEST(InternalOctree, irregular)
{
    internalOctreeIrregular<unsigned>();
    internalOctreeIrregular<uint64_t>();
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


//! \brief This creates an irregular tree. Checks geometry relations between children and parents.
template<class I>
void fullTreeIrregular()
{
    std::vector<I> codes = OctreeMaker<I>{}.divide().divide(0).divide(0,2).divide(3).makeTree();

    Octree<I> tree;
    tree.compute(codes.data(), codes.data() + codes.size()-1, 1);
    EXPECT_EQ(tree.nTreeNodes(), 33);
    EXPECT_EQ(tree.nLeaves(), 29);
    EXPECT_EQ(tree.nInternalNodes(), 4);

    Box<double> box(0,1);

    for (int i = 0; i < tree.nTreeNodes(); ++i)
    {
        if (i < 4)
            EXPECT_FALSE(tree.isLeaf(i));

        if (i == 0)
            EXPECT_TRUE(tree.isRoot(i));

        auto x = tree.x(i, box);
        auto y = tree.y(i, box);
        auto z = tree.z(i, box);

        // check geometrical relation between node i and its children
        if (!tree.isLeaf(i))
        {
            for (int hx = 0; hx < 2; ++hx)
            {
                for (int hy = 0; hy < 2; ++hy)
                {
                    for (int hz = 0; hz < 2; ++hz)
                    {
                        int octant = 4*hx + 2*hy + hz;
                        TreeNodeIndex child = tree.child(i, octant);

                        auto xChild = tree.x(child, box);
                        auto yChild = tree.y(child, box);
                        auto zChild = tree.z(child, box);

                        EXPECT_EQ(x[0], xChild[0] - hx * (xChild[1] - xChild[0]));
                        EXPECT_EQ(y[0], yChild[0] - hy * (yChild[1] - yChild[0]));
                        EXPECT_EQ(z[0], zChild[0] - hz * (zChild[1] - zChild[0]));

                        EXPECT_EQ(x[1], xChild[1] + (hx+1)%2 * (xChild[1] - xChild[0]));
                        EXPECT_EQ(y[1], yChild[1] + (hy+1)%2 * (yChild[1] - yChild[0]));
                        EXPECT_EQ(z[1], zChild[1] + (hz+1)%2 * (zChild[1] - zChild[0]));
                    }
                }
            }
        }

        // check geometrical relation between node i and its parent
        if (!tree.isRoot(i))
        {
            TreeNodeIndex parent = tree.parent(i);
            auto xParent = tree.x(parent, box);
            auto yParent = tree.y(parent, box);
            auto zParent = tree.z(parent, box);

            EXPECT_EQ(xParent[1] - xParent[0], 2 * (x[1] - x[0]));
            EXPECT_EQ(yParent[1] - yParent[0], 2 * (y[1] - y[0]));
            EXPECT_EQ(zParent[1] - zParent[0], 2 * (z[1] - z[0]));

            EXPECT_TRUE(xParent[0] <= x[0] && xParent[1] >= x[1]);
            EXPECT_TRUE(yParent[0] <= y[0] && yParent[1] >= y[1]);
            EXPECT_TRUE(zParent[0] <= z[0] && zParent[1] >= z[1]);
        }
    }

    //for (int i = 0; i < tree.nTreeNodes(); ++i)
    //{
    //    printf("node %3d, level %d, prefix %10o, parent %3d, ", i, tree.level(i), tree.codeStart(i), tree.parent(i));
    //    auto x = tree.x(i, box);
    //    auto y = tree.y(i, box);
    //    auto z = tree.z(i, box);
    //    printf("x: [%.4f:%4f], y: [%.4f:%.4f], z: [%.4f:%.4f]", x[0], x[1], y[0], y[1], z[0], z[1]);

    //    if (!tree.isLeaf(i))
    //    {
    //        std::cout << ", children: ";
    //        for (int octant = 0; octant < 8; ++octant)
    //            std::cout << tree.child(i, octant) << " ";
    //    }
    //    std::cout << std::endl;
    //}
}

TEST(InternalOctree, fullTreeIrregular)
{
    fullTreeIrregular<unsigned>();
    fullTreeIrregular<uint64_t>();
}