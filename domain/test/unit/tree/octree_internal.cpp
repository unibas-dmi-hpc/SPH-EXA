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
    using KeyType = unsigned;

    OctreeNode<KeyType> node1{0, 0, 0, {1, 2, 3, 4, 5, 6, 7, 8}};
    OctreeNode<KeyType> node2{0, 0, 0, {1, 2, 3, 4, 5, 6, 7, 8}};

    EXPECT_EQ(node1, node2);

    node2 = OctreeNode<KeyType>{1, 0, 0, {1, 2, 3, 4, 5, 6, 7, 8}};
    EXPECT_FALSE(node1 == node2);

    node2 = OctreeNode<KeyType>{0, 1, 0, {1, 2, 3, 4, 5, 6, 7, 8}};
    EXPECT_FALSE(node1 == node2);

    node2 = OctreeNode<KeyType>{0, 1, 0, {1, 2, 3, 4, 5, 6, 7, 8}};
    EXPECT_FALSE(node1 == node2);

    node2 = OctreeNode<KeyType>{0, 0, 1, {1, 2, 3, 4, 5, 6, 7, 8}};
    EXPECT_FALSE(node1 == node2);

    node2 = OctreeNode<KeyType>{0, 0, 0, {0, 2, 3, 4, 5, 6, 7, 8}};
    EXPECT_FALSE(node1 == node2);

    node2 = OctreeNode<KeyType>{0, 0, 0, {1, 2, 3, 4, 5, 6, 7, 9}};
    EXPECT_FALSE(node1 == node2);

    node2 = OctreeNode<KeyType>{0, 0, 0, {storeLeafIndex(1), 2, 3, 4, 5, 6, 7, 8}};
    EXPECT_FALSE(node1 == node2);

    node2 = OctreeNode<KeyType>{0, 0, 0, {1, 2, 3, 4, 5, 6, 7, storeLeafIndex(8)}};
    EXPECT_FALSE(node1 == node2);
}

/*! @brief test nodeDepth on a simple, explicitly constructed example
 *
 * Note: this example is not big enough to detect multithreading bugs, if present
 */
TEST(InternalOctree, nodeDepth)
{
    using KeyType = unsigned;
    constexpr auto& l_ = storeLeafIndex;

    // internal tree, matches leaves for
    // std::vector<KeyType> tree = OctreeMaker<KeyType>{}.divide().divide(0).divide(0,2).divide(3).makeTree();
    // clang-format off
    std::vector<OctreeNode<KeyType>> internalTree
    {
        {          0, 0, 0, {2, l_(19), l_(20), 3, l_(29), l_(30), l_(31), l_(32), }  },
        { 0200000000, 2, 2, {l_(6), l_(7), l_(8), l_(9), l_(10), l_(11), l_(12), l_(13), }    },
        {          0, 1, 0, {l_(4), l_(5), 1, l_(14), l_(15), l_(16), l_(17), l_(18), }   },
        {03000000000, 1, 0, {l_(21), l_(22), l_(23), l_(24), l_(25), l_(26), l_(27), l_(28), }}
    };
    // clang-format on

    std::vector<std::atomic<int>> depths(internalTree.size());
    for (auto& d : depths)
        d = 0;

    nodeDepth(internalTree.data(), TreeNodeIndex(internalTree.size()), depths.data());

    std::vector<int> depths_v{begin(depths), end(depths)};
    std::vector<int> depths_reference{3, 1, 2, 1};

    EXPECT_EQ(depths_v, depths_reference);
}

TEST(InternalOctree, topsort)
{
    using KeyType = uint32_t;

    std::vector<KeyType> leaves = makeUniformNLevelTree<KeyType>(512, 1);

    std::vector<BinaryNode<KeyType>> binaryTree(nNodes(leaves));
    createBinaryTree(leaves.data(), nNodes(leaves), binaryTree.data());

    TreeNodeIndex numLeafNodes = nNodes(leaves);
    // we ignore the last binary tree node which is a duplicate root node
    TreeNodeIndex nBinaryNodes = numLeafNodes - 1;

    // one extra element to store the total sum of the exclusive scan
    std::vector<TreeNodeIndex> binaryToOct(nBinaryNodes + 1);
    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < nBinaryNodes; ++i)
    {
        int  prefixLength = decodePrefixLength(binaryTree[i].prefix);
        bool divisibleBy3 = prefixLength % 3 == 0;
        binaryToOct[i] = (divisibleBy3) ? 1 : 0;
    }

    // stream compaction: scan and scatter
    exclusiveScan(binaryToOct.data(), binaryToOct.size());

    // nInternalOctreeNodes is also equal to prefixes[nBinaryNodes]
    TreeNodeIndex numInternalNodes = (numLeafNodes - 1) / 7;
    std::vector<TreeNodeIndex> octToBinary(numInternalNodes);

    // compaction step, scatterMap -> compacted list of binary nodes that correspond to octree nodes
    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < nBinaryNodes; ++i)
    {
        bool isOctreeNode = (binaryToOct[i+1] - binaryToOct[i]) == 1;
        if (isOctreeNode)
        {
            int octreeNodeIndex = binaryToOct[i];
            octToBinary[octreeNodeIndex] = i;
        }
    }

    TreeNodeIndex numNodes = numInternalNodes + numLeafNodes;
    std::vector<KeyType> prefixes(numNodes);
    std::vector<unsigned> levels(numNodes);
    std::vector<TreeNodeIndex> nodeOrder(numNodes);
    std::vector<TreeNodeIndex> levelOffsets(maxTreeLevel<KeyType>{} + 2);
    std::iota(begin(nodeOrder), end(nodeOrder), 0);

    for (TreeNodeIndex i = 0; i < numInternalNodes; ++i)
    {
        TreeNodeIndex binaryIndex = octToBinary[i];
        levels[i]   = decodePrefixLength(binaryTree[binaryIndex].prefix) / 3;
        prefixes[i] = decodePlaceholderBit(binaryTree[binaryIndex].prefix);
    }

    std::copy(begin(leaves), begin(leaves) + numLeafNodes, begin(prefixes) + numInternalNodes);

    // compute leaf node tree levels
    for (TreeNodeIndex i = 0; i < numLeafNodes; ++i)
    {
        unsigned level = treeLevel(leaves[i+1] - leaves[i]);
        levels[i + numInternalNodes] = level;
    }

    /*! prefix and levels now in unsorted layout A
     *
     * binaryTree |---------------------------------------------------|
     *                    ^                      |
     *   octToBinary   |--|  |-------------------|  binaryToOct
     *                 |     |
     * prefixes   |------------|--------------------------------|
     * levels     |------------|--------------------------------|
     *              internal        leaves
     */

    sort_by_key(begin(levels), end(levels), begin(nodeOrder));
    reorderInPlace(nodeOrder, prefixes.data());

    for (unsigned level = 0; level < maxTreeLevel<KeyType>{} + 1; ++level)
    {
        auto it1 = std::lower_bound(begin(levels), end(levels), level);
        auto it2 = std::upper_bound(begin(levels), end(levels), level);
        levelOffsets[level] = it2 - it1;
    }

    exclusiveScan(levelOffsets.data(), levelOffsets.size());

    for (unsigned level = 0; level < maxTreeLevel<KeyType>{} + 1; ++level)
    {
        //std::cout << level << " levelSize " << levelOffsets[level] << std::endl;
        TreeNodeIndex lvlStart = levelOffsets[level];
        sort_by_key(begin(prefixes) + lvlStart, begin(prefixes) + levelOffsets[level + 1], begin(nodeOrder) + lvlStart);
    }

    /*! prefix and levels now in sorted layout B
     *
     *  -levels is sorted in ascending order
     *  -prefix is first sorted by level, then by ascening key
     *  -nodeOrder goes from layout B to layout A (nodeOrder[i] is i's location in A)
     */

    std::vector<TreeNodeIndex> inverseNodeOrder(numNodes);
    std::iota(begin(inverseNodeOrder), end(inverseNodeOrder), 0);

    // compute inverse of nodeOrder, invalidates nodeOrder
    sort_by_key(begin(nodeOrder), end(nodeOrder), begin(inverseNodeOrder));

    std::vector<TreeNodeIndex> childOffsets(numNodes, 0);
    std::vector<TreeNodeIndex> parents( (numNodes - 1) / 8);

    // loop over octree nodes index in layout A
    for (TreeNodeIndex idxA = 0; idxA < numInternalNodes; ++idxA)
    {
        TreeNodeIndex binaryIndex = octToBinary[idxA];
        TreeNodeIndex firstChild  = binaryTree[binaryTree[binaryTree[binaryIndex].child[0]].child[0]].child[0];

        // octree node index in layout B
        TreeNodeIndex idxB = inverseNodeOrder[idxA];

        // child node index in layout A
        TreeNodeIndex childA =
            (isLeafIndex(firstChild)) ? loadLeafIndex(firstChild) + numInternalNodes : binaryToOct[firstChild];

        // node index in layout B
        TreeNodeIndex childB = inverseNodeOrder[childA];

        childOffsets[idxB]        = childB;
        parents[(childB - 1) / 8] = idxB;
    }

    TreeNodeIndex numLeavesControl = std::count(begin(childOffsets), end(childOffsets), 0);
    EXPECT_EQ(numLeavesControl, numLeafNodes);

    for (TreeNodeIndex i = 0; i < numNodes; ++i)
    {
        TreeNodeIndex childIdx = childOffsets[i];
        EXPECT_TRUE(childIdx == 0 || levels[childIdx] == levels[i] + 1);

        if (childIdx != 0)
        {
            EXPECT_EQ(i, parents[(childIdx - 1) / 8]);
            for (int octant = 0; octant < 8; ++octant)
            {
                EXPECT_EQ(prefixes[i] + octant * nodeRange<KeyType>(levels[childIdx]), prefixes[childIdx + octant]);
            }
        }
    }
}


/*! @brief larger test case for nodeDepth to detect multithreading issues
 *
 * Depends on binary/octree generation, so not strictly a unit test
 */
template<class KeyType>
void nodeDepthThreading()
{
    // uniform 16x16x16 tree
    std::vector<KeyType> leaves = makeUniformNLevelTree<KeyType>(4096, 1);

    std::vector<BinaryNode<KeyType>> binaryTree(nNodes(leaves));
    createBinaryTree(leaves.data(), nNodes(leaves), binaryTree.data());

    std::vector<OctreeNode<KeyType>> octree((nNodes(leaves) - 1) / 7);
    std::vector<TreeNodeIndex> leafParents(nNodes(leaves));

    createInternalOctreeCpu(binaryTree.data(), nNodes(leaves), octree.data(), leafParents.data());

    std::vector<std::atomic<int>> depths(octree.size());
    for (auto& d : depths)
        d = 0;

    nodeDepth(octree.data(), octree.size(), depths.data());
    std::vector<int> depths_v{begin(depths), end(depths)};

    constexpr int maxTreeLevel = 4; // tree has 4 layers of subdivisions
    std::vector<int> depths_reference(octree.size());
    for (TreeNodeIndex i = 0; i < TreeNodeIndex(octree.size()); ++i)
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
    using KeyType = unsigned;
    constexpr auto& l_ = storeLeafIndex;

    // internal tree, matches leaves for
    // std::vector<KeyType> tree = OctreeMaker<KeyType>{}.divide().divide(0).divide(0,2).divide(3).makeTree();
    // clang-format off
    std::vector<OctreeNode<KeyType>> octree
    {
        // prefix, level, parent, children
        {          0, 0, 0, {2, l_(19), l_(20), 3, l_(29), l_(30), l_(31), l_(32), }  },
        { 0200000000, 2, 2, {l_(6), l_(7), l_(8), l_(9), l_(10), l_(11), l_(12), l_(13), }    },
        {          0, 1, 0, {l_(4), l_(5), 1, l_(14), l_(15), l_(16), l_(17), l_(18), }   },
        {03000000000, 1, 0, {l_(21), l_(22), l_(23), l_(24), l_(25), l_(26), l_(27), l_(28), }}
    };
    // clang-format on

    std::vector<TreeNodeIndex> ordering(octree.size());
    std::vector<TreeNodeIndex> nNodesPerLevel(maxTreeLevel<KeyType>{});
    decreasingMaxDepthOrder(octree.data(), octree.size(), ordering.data(), nNodesPerLevel.data());

    std::vector<TreeNodeIndex> reference{0, 2, 1, 3};
    EXPECT_EQ(ordering, reference);

    std::vector<TreeNodeIndex> nNodesPerLevelReference(maxTreeLevel<KeyType>{}, 0);
    nNodesPerLevelReference[1] = 2;
    nNodesPerLevelReference[2] = 1;
    nNodesPerLevelReference[3] = 1;
    EXPECT_EQ(nNodesPerLevel, nNodesPerLevelReference);
}

template<class KeyType>
void decreasingMaxDepthOrderIsSorted()
{
    // uniform 16x16x16 tree
    std::vector<KeyType> leaves = makeUniformNLevelTree<KeyType>(4096, 1);

    std::vector<BinaryNode<KeyType>> binaryTree(nNodes(leaves));
    createBinaryTree(leaves.data(), nNodes(leaves), binaryTree.data());

    std::vector<OctreeNode<KeyType>> octree((nNodes(leaves) - 1) / 7);
    std::vector<TreeNodeIndex> leafParents(nNodes(leaves));

    createInternalOctreeCpu(binaryTree.data(), nNodes(leaves), octree.data(), leafParents.data());

    std::vector<TreeNodeIndex> depthOrder(octree.size());
    std::vector<TreeNodeIndex> nNodesPerLevel(maxTreeLevel<KeyType>{}, 0);
    decreasingMaxDepthOrder(octree.data(), octree.size(), depthOrder.data(), nNodesPerLevel.data());

    std::vector<OctreeNode<KeyType>> newOctree(octree.size());
    rewireInternal(octree.data(), depthOrder.data(), octree.size(), newOctree.data());

    std::vector<std::atomic<int>> depths(octree.size());
    for (auto& d : depths)
        d = 0;

    nodeDepth(newOctree.data(), newOctree.size(), depths.data());
    std::vector<int> depths_v{begin(depths), end(depths)};

    EXPECT_TRUE(std::is_sorted(begin(depths_v), end(depths_v), std::greater<TreeNodeIndex>{}));

    std::vector<TreeNodeIndex> nNodesPerLevelReference(maxTreeLevel<KeyType>{}, 0);
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

template<class KeyType>
void checkConnectivity(const Octree<KeyType>& fullTree)
{
    ASSERT_TRUE(fullTree.isRoot(0));

    // check all internal nodes
    for (TreeNodeIndex nodeIdx = 0; nodeIdx < fullTree.numInternalNodes(); ++nodeIdx)
    {
        ASSERT_FALSE(fullTree.isLeaf(nodeIdx));

        KeyType prefix = fullTree.codeStart(nodeIdx);
        int level = fullTree.level(nodeIdx);

        EXPECT_EQ(fullTree.codeEnd(nodeIdx), prefix + nodeRange<KeyType>(level));

        for (int octant = 0; octant < 8; ++octant)
        {
            TreeNodeIndex child = fullTree.child(nodeIdx, octant);
            EXPECT_EQ(prefix + octant * nodeRange<KeyType>(level + 1), fullTree.codeStart(child));
        }

        if (!fullTree.isRoot(nodeIdx))
        {
            TreeNodeIndex parent = fullTree.parent(nodeIdx);
            EXPECT_EQ(fullTree.level(parent), level - 1);

            KeyType parentPrefix = fullTree.codeStart(parent);
            EXPECT_EQ(parentPrefix, enclosingBoxCode(prefix, level - 1));
        }
        else
        {
            EXPECT_EQ(fullTree.codeStart(nodeIdx), 0);
            EXPECT_EQ(fullTree.level(nodeIdx), 0);
        }
    }

    // check all leaf nodes
    for (TreeNodeIndex nodeIdx = fullTree.numInternalNodes(); nodeIdx < fullTree.numTreeNodes(); ++nodeIdx)
    {
        ASSERT_TRUE(fullTree.isLeaf(nodeIdx));

        KeyType prefix = fullTree.codeStart(nodeIdx);
        int level = fullTree.level(nodeIdx);

        EXPECT_EQ(fullTree.codeEnd(nodeIdx), prefix + nodeRange<KeyType>(level));

        TreeNodeIndex parent = fullTree.parent(nodeIdx);
        EXPECT_EQ(fullTree.level(parent), level - 1);

        KeyType parentPrefix = fullTree.codeStart(parent);
        EXPECT_EQ(parentPrefix, enclosingBoxCode(prefix, level - 1));
    }
}

TEST(InternalOctree, rewire)
{
    using KeyType = unsigned;
    constexpr auto& l_ = storeLeafIndex;

    // internal tree, matches leaves for
    // std::vector<KeyType> tree = OctreeMaker<KeyType>{}.divide().divide(0).divide(0,2).divide(3).makeTree();
    // clang-format off
    std::vector<OctreeNode<KeyType>> internalTree
    {
        // prefix, level, parent, children, childTypes
        {          0, 0, 0, {2, l_(19), l_(20), 3, l_(29), l_(30), l_(31), l_(32), }  },
        { 0200000000, 2, 2, {l_(6), l_(7), l_(8), l_(9), l_(10), l_(11), l_(12), l_(13), }    },
        {          0, 1, 0, {l_(4), l_(5), 1, l_(14), l_(15), l_(16), l_(17), l_(18), }   },
        {03000000000, 1, 0, {l_(21), l_(22), l_(23), l_(24), l_(25), l_(26), l_(27), l_(28), }}
    };
    // clang-format on


    // maps oldIndex to rewireMap[oldIndex] (scatter operation)
    std::vector<TreeNodeIndex> rewireMap{0, 3, 1, 2};

    std::vector<OctreeNode<KeyType>> rewiredTree(internalTree.size());
    rewireInternal(internalTree.data(), rewireMap.data(), TreeNodeIndex(internalTree.size()), rewiredTree.data());

    //for (int i = 0; i < rewiredTree.size(); ++i)
    //{
    //    printf("node %3d, prefix %10o, level %1d\n", i, rewiredTree[i].prefix, rewiredTree[i].level);
    //}

    // clang-format off
    std::vector<OctreeNode<KeyType>> reference
    {
        // prefix, level, parent, children
        {          0, 0, 0, {1, l_(19), l_(20), 2, l_(29), l_(30), l_(31), l_(32), }  },
        {          0, 1, 0, {l_(4), l_(5), 3, l_(14), l_(15), l_(16), l_(17), l_(18), }   },
        {03000000000, 1, 0, {l_(21), l_(22), l_(23), l_(24), l_(25), l_(26), l_(27), l_(28), }},
        { 0200000000, 2, 1, {l_(6), l_(7), l_(8), l_(9), l_(10), l_(11), l_(12), l_(13), }    }
    };
    // clang-format on

    EXPECT_EQ(rewiredTree, reference);
}

TEST(InternalOctree, rootNode)
{
    auto tree = makeRootNodeTree<unsigned>();

    Octree<unsigned> fullTree;
    fullTree.update(tree.data(), tree.data() + tree.size());

    EXPECT_EQ(fullTree.numLeafNodes(), 1);
    EXPECT_EQ(fullTree.numTreeNodes(), 1);
    EXPECT_EQ(fullTree.numInternalNodes(), 0);
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
template<class KeyType>
void octree4x4x4()
{
    std::vector<KeyType> tree = makeUniformNLevelTree<KeyType>(64, 1);

    Octree<KeyType> fullTree;
    fullTree.update(tree.data(), tree.data() + tree.size());

    ASSERT_EQ(fullTree.numInternalNodes(), (64 - 1) / 7);
    ASSERT_EQ(fullTree.numLeafNodes(), 64);

    EXPECT_EQ(fullTree.numTreeNodes(0), 64);
    EXPECT_EQ(fullTree.numTreeNodes(1), 8);
    EXPECT_EQ(fullTree.numTreeNodes(2), 1);

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
template<class KeyType>
void octreeIrregularL2()
{
    std::vector<KeyType> tree = OctreeMaker<KeyType>{}.divide().divide(0).makeTree();

    Octree<KeyType> fullTree;
    fullTree.update(tree.data(), tree.data() + tree.size());

    ASSERT_EQ(fullTree.numInternalNodes(), (15 - 1) / 7);
    ASSERT_EQ(fullTree.numLeafNodes(), 15);

    EXPECT_EQ(fullTree.numTreeNodes(0), 15);
    EXPECT_EQ(fullTree.numTreeNodes(1), 1);
    EXPECT_EQ(fullTree.numTreeNodes(2), 1);

    checkConnectivity(fullTree);
}

TEST(InternalOctree, irregularL2)
{
    octreeIrregularL2<unsigned>();
    octreeIrregularL2<uint64_t>();
}

//! @brief This creates an irregular tree. Checks geometry relations between children and parents.
template<class KeyType>
void octreeIrregularL3()
{
    std::vector<KeyType> tree = OctreeMaker<KeyType>{}.divide().divide(0).divide(0, 2).divide(3).makeTree();

    Octree<KeyType> fullTree;
    fullTree.update(tree.data(), tree.data() + tree.size());
    EXPECT_EQ(fullTree.numTreeNodes(), 33);
    EXPECT_EQ(fullTree.numLeafNodes(), 29);
    EXPECT_EQ(fullTree.numInternalNodes(), 4);

    EXPECT_EQ(fullTree.numTreeNodes(0), 29);
    EXPECT_EQ(fullTree.numTreeNodes(1), 2);
    EXPECT_EQ(fullTree.numTreeNodes(2), 1);
    EXPECT_EQ(fullTree.numTreeNodes(3), 1);

    checkConnectivity(fullTree);
}

TEST(InternalOctree, irregularL3)
{
    octreeIrregularL3<unsigned>();
    octreeIrregularL3<uint64_t>();
}

template<class KeyType>
void locateTest()
{
    std::vector<KeyType> cornerstones{0, 1, nodeRange<KeyType>(0) - 1, nodeRange<KeyType>(0)};
    std::vector<KeyType> spanningTree = computeSpanningTree<KeyType>(cornerstones);

    Octree<KeyType> fullTree;
    fullTree.update(std::move(spanningTree));

    std::vector<std::array<KeyType, 2>> inputs{{0, nodeRange<KeyType>(0)},
                                               {0, nodeRange<KeyType>(1)},
                                               {0, nodeRange<KeyType>(2)},
                                               {0, nodeRange<KeyType>(3)},
                                               {0, 1},
                                               {4 * nodeRange<KeyType>(1), 5 * nodeRange<KeyType>(1)},
                                               {nodeRange<KeyType>(0) - 512, nodeRange<KeyType>(0)},
                                               {nodeRange<KeyType>(0) - 64, nodeRange<KeyType>(0)},
                                               {nodeRange<KeyType>(0) - 8, nodeRange<KeyType>(0)},
                                               {nodeRange<KeyType>(0) - 1, nodeRange<KeyType>(0)}};

    for (auto p : inputs)
    {
        auto [start, end] = p;
        TreeNodeIndex nodeIdx = fullTree.locate(start, end);
        EXPECT_EQ(start, fullTree.codeStart(nodeIdx));
        EXPECT_EQ(end, fullTree.codeEnd(nodeIdx));
    }

    EXPECT_EQ(fullTree.locate(0, 2), fullTree.numTreeNodes());
}

TEST(InternalOctree, locate)
{
    locateTest<unsigned>();
    locateTest<uint64_t>();
}
