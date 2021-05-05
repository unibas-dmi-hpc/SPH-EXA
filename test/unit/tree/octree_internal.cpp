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

/*! @file
 * @brief octree utility tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * This file implements tests for OctreeMaker.
 * OctreeMaker can be used to generate octrees in cornerstone
 * format. It is only used to test the octree implementation.
 */

#include "gtest/gtest.h"

#include "cstone/tree/octree_internal.hpp"
#include "cstone/tree/octree_util.hpp"

using namespace cstone;

//! @brief test OctreNode equality comparison
TEST(InternalOctree, OctreeNodeEq)
{
    using I = unsigned;

    OctreeNode<I> node1{0, 0, 0, {1,2,3,4,5,6,7,8}};
    OctreeNode<I> node2{0, 0, 0, {1,2,3,4,5,6,7,8}};

    EXPECT_EQ(node1, node2);

    node2 = OctreeNode<I>{1, 0, 0, {1,2,3,4,5,6,7,8}};
    EXPECT_FALSE(node1 == node2);

    node2 = OctreeNode<I>{0, 1, 0, {1,2,3,4,5,6,7,8}};
    EXPECT_FALSE(node1 == node2);

    node2 = OctreeNode<I>{0, 1, 0, {1,2,3,4,5,6,7,8}};
    EXPECT_FALSE(node1 == node2);

    node2 = OctreeNode<I>{0, 0, 1, {1,2,3,4,5,6,7,8}};
    EXPECT_FALSE(node1 == node2);

    node2 = OctreeNode<I>{0, 0, 0, {0,2,3,4,5,6,7,8}};
    EXPECT_FALSE(node1 == node2);

    node2 = OctreeNode<I>{0, 0, 0, {1,2,3,4,5,6,7,9}};
    EXPECT_FALSE(node1 == node2);

    node2 = OctreeNode<I>{0, 0, 0, {storeLeafIndex(1),2,3,4,5,6,7,8}};
    EXPECT_FALSE(node1 == node2);

    node2 = OctreeNode<I>{0, 0, 0, {1,2,3,4,5,6,7,storeLeafIndex(8)}};
    EXPECT_FALSE(node1 == node2);
}

/*! @brief test nodeDepth on a simple, explicitly constructed example
 *
 * Note: this example is not big enough to detect multithreading bugs, if present
 */
TEST(InternalOctree, nodeDepth)
{
    using I = unsigned;
    constexpr auto& l_ = storeLeafIndex;

    // internal tree, matches leaves for
    // std::vector<I> tree = OctreeMaker<I>{}.divide().divide(0).divide(0,2).divide(3).makeTree();
    std::vector<OctreeNode<I>> internalTree
    {
        {          0, 0, 0, {2, l_(19), l_(20), 3, l_(29), l_(30), l_(31), l_(32), }  },
        { 0200000000, 2, 2, {l_(6), l_(7), l_(8), l_(9), l_(10), l_(11), l_(12), l_(13), }    },
        {          0, 1, 0, {l_(4), l_(5), 1, l_(14), l_(15), l_(16), l_(17), l_(18), }   },
        {03000000000, 1, 0, {l_(21), l_(22), l_(23), l_(24), l_(25), l_(26), l_(27), l_(28), }}
    };

    std::vector<std::atomic<int>> depths(internalTree.size());
    for (auto& d : depths) d = 0;

    nodeDepth(internalTree.data(), TreeNodeIndex(internalTree.size()), depths.data());

    std::vector<int> depths_v{begin(depths), end(depths)};
    std::vector<int> depths_reference{3, 1, 2, 1};

    EXPECT_EQ(depths_v, depths_reference);
}

/*! @brief larger test case for nodeDepth to detect multithreading issues
 *
 * Depends on binary/octree generation, so not strictly a unit test
 */
template<class I>
void nodeDepthThreading()
{
    // uniform 16x16x16 tree
    std::vector<I> leaves = makeUniformNLevelTree<I>(4096, 1);

    std::vector<BinaryNode<I>> binaryTree(nNodes(leaves));
    createBinaryTree(leaves.data(), nNodes(leaves), binaryTree.data());

    std::vector<OctreeNode<I>> octree((nNodes(leaves)-1)/7);
    std::vector<TreeNodeIndex> leafParents(nNodes(leaves));

    createInternalOctreeCpu(binaryTree.data(), nNodes(leaves), octree.data(), leafParents.data());

    std::vector<std::atomic<int>> depths(octree.size());
    for (auto& d : depths) d = 0;

    nodeDepth(octree.data(), octree.size(), depths.data());
    std::vector<int> depths_v{begin(depths), end(depths)};

    constexpr int maxTreeLevel = 4;         // tree has 4 layers of subdivisions
    std::vector<int> depths_reference(octree.size());
    for (TreeNodeIndex i = 0; i < octree.size(); ++i)
    {
        // in a uniform tree, level + depth == maxTreeLevel is constant for all nodes
        depths_reference[i] = maxTreeLevel - octree[i].level;
    }

    EXPECT_EQ(depths_v, depths_reference);
}

TEST(InternalOctree, nodeDepthsThreading)
{
    nodeDepthThreading<unsigned>();
    nodeDepthThreading<uint64_t>();
}

TEST(InternalOctree, calculateInternalOrderExplicit)
{
    using I = unsigned;
    constexpr auto& l_ = storeLeafIndex;

    // internal tree, matches leaves for
    // std::vector<I> tree = OctreeMaker<I>{}.divide().divide(0).divide(0,2).divide(3).makeTree();
    std::vector<OctreeNode<I>> octree
    {
        // prefix, level, parent, children
        {          0, 0, 0, {2, l_(19), l_(20), 3, l_(29), l_(30), l_(31), l_(32), }  },
        { 0200000000, 2, 2, {l_(6), l_(7), l_(8), l_(9), l_(10), l_(11), l_(12), l_(13), }    },
        {          0, 1, 0, {l_(4), l_(5), 1, l_(14), l_(15), l_(16), l_(17), l_(18), }   },
        {03000000000, 1, 0, {l_(21), l_(22), l_(23), l_(24), l_(25), l_(26), l_(27), l_(28), }}
    };

    std::vector<TreeNodeIndex> ordering(octree.size());
    std::vector<TreeNodeIndex> nNodesPerLevel(maxTreeLevel<I>{});
    decreasingMaxDepthOrder(octree.data(), octree.size(), ordering.data(), nNodesPerLevel.data());

    std::vector<TreeNodeIndex> reference{0, 2, 1, 3};
    EXPECT_EQ(ordering, reference);

    std::vector<TreeNodeIndex> nNodesPerLevelReference(maxTreeLevel<I>{}, 0);
    nNodesPerLevelReference[1] = 2;
    nNodesPerLevelReference[2] = 1;
    nNodesPerLevelReference[3] = 1;
    EXPECT_EQ(nNodesPerLevel, nNodesPerLevelReference);
}

template<class I>
void decreasingMaxDepthOrderIsSorted()
{
    // uniform 16x16x16 tree
    std::vector<I> leaves = makeUniformNLevelTree<I>(4096, 1);

    std::vector<BinaryNode<I>> binaryTree(nNodes(leaves));
    createBinaryTree(leaves.data(), nNodes(leaves), binaryTree.data());

    std::vector<OctreeNode<I>> octree((nNodes(leaves)-1)/7);
    std::vector<TreeNodeIndex> leafParents(nNodes(leaves));

    createInternalOctreeCpu(binaryTree.data(), nNodes(leaves), octree.data(), leafParents.data());

    std::vector<TreeNodeIndex> depthOrder(octree.size());
    std::vector<TreeNodeIndex> nNodesPerLevel(maxTreeLevel<I>{}, 0);
    decreasingMaxDepthOrder(octree.data(), octree.size(), depthOrder.data(), nNodesPerLevel.data());

    std::vector<OctreeNode<I>> newOctree(octree.size());
    rewireInternal(octree.data(), depthOrder.data(), octree.size(), newOctree.data());

    std::vector<std::atomic<int>> depths(octree.size());
    for (auto& d : depths) d = 0;

    nodeDepth(newOctree.data(), newOctree.size(), depths.data());
    std::vector<int> depths_v{begin(depths), end(depths)};

    EXPECT_TRUE(std::is_sorted(begin(depths_v), end(depths_v), std::greater<TreeNodeIndex>{}));

    std::vector<TreeNodeIndex> nNodesPerLevelReference(maxTreeLevel<I>{}, 0);
    nNodesPerLevelReference[1] = 512;
    nNodesPerLevelReference[2] = 64;
    nNodesPerLevelReference[3] = 8;
    nNodesPerLevelReference[4] = 1;
    EXPECT_EQ(nNodesPerLevel, nNodesPerLevelReference);
}

TEST(InternalOctree, decreasingMaxDepthOrderIsSorted)
{
    decreasingMaxDepthOrderIsSorted<unsigned>();
    decreasingMaxDepthOrderIsSorted<uint64_t>();
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

TEST(InternalOctree, rewire)
{
    using I = unsigned;
    constexpr auto& l_ = storeLeafIndex;

    // internal tree, matches leaves for
    // std::vector<I> tree = OctreeMaker<I>{}.divide().divide(0).divide(0,2).divide(3).makeTree();
    std::vector<OctreeNode<I>> internalTree
    {
        // prefix, level, parent, children, childTypes
        {          0, 0, 0, {2, l_(19), l_(20), 3, l_(29), l_(30), l_(31), l_(32), }  },
        { 0200000000, 2, 2, {l_(6), l_(7), l_(8), l_(9), l_(10), l_(11), l_(12), l_(13), }    },
        {          0, 1, 0, {l_(4), l_(5), 1, l_(14), l_(15), l_(16), l_(17), l_(18), }   },
        {03000000000, 1, 0, {l_(21), l_(22), l_(23), l_(24), l_(25), l_(26), l_(27), l_(28), }}
    };

    // maps oldIndex to rewireMap[oldIndex] (scatter operation)
    std::vector<TreeNodeIndex> rewireMap{0,3,1,2};

    std::vector<OctreeNode<I>> rewiredTree(internalTree.size());
    rewireInternal(internalTree.data(), rewireMap.data(), TreeNodeIndex(internalTree.size()),
                   rewiredTree.data());

    for (int i = 0; i < rewiredTree.size(); ++i)
    {
        printf("node %3d, prefix %10o, level %1d\n", i, rewiredTree[i].prefix, rewiredTree[i].level);
    }

    std::vector<OctreeNode<I>> reference
    {
        // prefix, level, parent, children
        {          0, 0, 0, {1, l_(19), l_(20), 2, l_(29), l_(30), l_(31), l_(32), }  },
        {          0, 1, 0, {l_(4), l_(5), 3, l_(14), l_(15), l_(16), l_(17), l_(18), }   },
        {03000000000, 1, 0, {l_(21), l_(22), l_(23), l_(24), l_(25), l_(26), l_(27), l_(28), }},
        { 0200000000, 2, 1, {l_(6), l_(7), l_(8), l_(9), l_(10), l_(11), l_(12), l_(13), }    }
    };

    EXPECT_EQ(rewiredTree, reference);
}

TEST(InternalOctree, rootNode)
{
    auto tree = makeRootNodeTree<unsigned>();

    Octree<unsigned> fullTree;
    fullTree.update(tree.data(), tree.data() + tree.size());

    EXPECT_EQ(fullTree.nLeafNodes(), 1);
    EXPECT_EQ(fullTree.nTreeNodes(), 1);
    EXPECT_EQ(fullTree.nInternalNodes(), 0);
    EXPECT_EQ(fullTree.codeStart(0), 0);
    EXPECT_EQ(fullTree.codeEnd(0), nodeRange<unsigned>(0));
}

/*! @brief test internal octree creation from a regular 4x4x4 grid of leaves
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
    ASSERT_EQ(fullTree.nLeafNodes(), 64);

    EXPECT_EQ(fullTree.nTreeNodes(0), 64);
    EXPECT_EQ(fullTree.nTreeNodes(1), 8);
    EXPECT_EQ(fullTree.nTreeNodes(2), 1);

    checkConnectivity(fullTree);
}

TEST(InternalOctree, octree4x4x4)
{
    octree4x4x4<unsigned>();
    octree4x4x4<uint64_t>();
}

/*! @brief test internal octree creation with an irregular leaf tree
 *
 * The leaf tree is the result of subdividing the root node, then further
 * subdividing octant 0. This results in 15 leaves, so the internal tree
 * should have two nodes: the root and the one internal level-1 node for the
 * first octant. The root points to the one internal node and to leaves [8:15].
 * The internal level-1 nodes points to leaves [0:8].
 */
template<class I>
void octreeIrregularL2()
{
    std::vector<I> tree = OctreeMaker<I>{}.divide().divide(0).makeTree();

    Octree<I> fullTree;
    fullTree.update(tree.data(), tree.data() + tree.size());

    ASSERT_EQ(fullTree.nInternalNodes(), (15 - 1) / 7);
    ASSERT_EQ(fullTree.nLeafNodes(), 15);

    EXPECT_EQ(fullTree.nTreeNodes(0), 15);
    EXPECT_EQ(fullTree.nTreeNodes(1), 1);
    EXPECT_EQ(fullTree.nTreeNodes(2), 1);

    checkConnectivity(fullTree);
}

TEST(InternalOctree, irregularL2)
{
    octreeIrregularL2<unsigned>();
    octreeIrregularL2<uint64_t>();
}

//! @brief This creates an irregular tree. Checks geometry relations between children and parents.
template<class I>
void octreeIrregularL3()
{
    std::vector<I> tree = OctreeMaker<I>{}.divide().divide(0).divide(0,2).divide(3).makeTree();

    Octree<I> fullTree;
    fullTree.update(tree.data(), tree.data() + tree.size());
    EXPECT_EQ(fullTree.nTreeNodes(), 33);
    EXPECT_EQ(fullTree.nLeafNodes(), 29);
    EXPECT_EQ(fullTree.nInternalNodes(), 4);

    EXPECT_EQ(fullTree.nTreeNodes(0), 29);
    EXPECT_EQ(fullTree.nTreeNodes(1), 2);
    EXPECT_EQ(fullTree.nTreeNodes(2), 1);
    EXPECT_EQ(fullTree.nTreeNodes(3), 1);

    checkConnectivity(fullTree);
}

TEST(InternalOctree, irregularL3)
{
    octreeIrregularL3<unsigned>();
    octreeIrregularL3<uint64_t>();
}