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
 * @brief Test locally essential octree
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/focus/octree_focus.hpp"
#include "cstone/tree/cs_util.hpp"

namespace cstone
{

template<class KeyType>
TreeNodeIndex numNodesInRange(gsl::span<const KeyType> tree, KeyType a, KeyType b)
{
    auto itb = std::lower_bound(tree.begin(), tree.end(), b);
    auto ita = std::lower_bound(tree.begin(), tree.end(), a);

    EXPECT_EQ(*itb, b);
    EXPECT_EQ(*ita, a);

    return itb - ita;
}

template<class KeyType>
std::vector<TreeNodeIndex> octantNodeCount(gsl::span<const KeyType> tree)
{
    std::vector<TreeNodeIndex> counts;
    for (int octant = 0; octant < 8; ++octant)
    {
        auto range = nodeRange<KeyType>(1);
        counts.push_back(numNodesInRange(tree, octant * range, octant * range + range));
    }

    std::sort(begin(counts), end(counts));

    // omit the last octant with the highest node count (the focused octant)
    return {counts.begin(), counts.begin() + 7};
}

template<class KeyType>
static void computeEssentialTree()
{
    Box<double> box{-1, 1};
    int nParticles        = 200000;
    unsigned csBucketSize = 16;

    std::vector<KeyType> codes(nParticles);
    std::iota(codes.begin(), codes.end(), 0);
    std::for_each(codes.begin(), codes.end(), [n = nParticles](auto& k) { k *= double(nodeRange<KeyType>(0)) / n; });

    auto [csTree, csCounts] = computeOctree(codes.data(), codes.data() + nParticles, csBucketSize);
    Octree<KeyType> globalTree;
    globalTree.update(csTree.data(), nNodes(csTree));

    unsigned bucketSize = 16;
    float theta         = 0.9;
    FocusedOctreeSingleNode<KeyType> tree(bucketSize, theta);

    // sorted reference tree node counts in each (except focus) octant at the 1st division level
    std::vector<TreeNodeIndex> refCounts{29, 302, 302, 302, 1184, 1184, 1184,
                                         /*4131*/};

    KeyType focusStart = 1;
    KeyType focusEnd   = pad(KeyType(1), 3);

    tree.update(box, codes, focusStart, focusEnd, {});
    // The focus boundaries have to be contained in the tree, even after just one update step.
    // This example here is the worst-case scenario with a focus boundary at the highest possible
    // octree subdivision level. Key-0 is always present, so the node with Key-1 is always at index 1, if present.
    EXPECT_EQ(tree.treeLeaves()[1], focusStart);

    // update until converged
    while (!tree.update(box, codes, focusStart, focusEnd, {})) {}

    {
        // the first node in the cornerstone tree that starts at or above focusStart
        TreeNodeIndex firstCstoneNode = findNodeAbove(csTree.data(), csTree.size(), focusStart);
        TreeNodeIndex matchingFocusNode =
            findNodeAbove(tree.treeLeaves().data(), tree.treeLeaves().size(), csTree[firstCstoneNode]);
        // in the focus area (the first octant) the essential tree and the csTree are identical
        TreeNodeIndex lastFocusNode = findNodeAbove(tree.treeLeaves().data(), tree.treeLeaves().size(), focusEnd);

        EXPECT_TRUE(matchingFocusNode >= 1 && matchingFocusNode < lastFocusNode);

        // We have: focusTree[matchingFocusNode] == csTree[firstCstoneNode]
        // therefore all nodes past matchingFocusNode until lastFocusNode should match those in the csTree
        EXPECT_TRUE(std::equal(tree.treeLeaves().begin() + matchingFocusNode, tree.treeLeaves().begin() + lastFocusNode,
                               begin(csTree) + firstCstoneNode));

        // From 0 to matchingFocusNode, the focusTree should be identical to the spanningTree
        std::vector<KeyType> spanningKeys{0, focusStart, nodeRange<KeyType>(0)};
        auto spanningTree = computeSpanningTree<KeyType>(spanningKeys);
        EXPECT_EQ(tree.treeLeaves().first(matchingFocusNode),
                  gsl::span<KeyType>(spanningTree.data(), matchingFocusNode));

        auto nodeCounts = octantNodeCount<KeyType>(tree.treeLeaves());
        EXPECT_EQ(nodeCounts, refCounts);
    }

    focusStart = pad(KeyType(7), 3);
    focusEnd   = nodeRange<KeyType>(0) - 1;
    while (!tree.update(box, codes, focusStart, focusEnd, {})) {}

    {
        auto nodeCounts = octantNodeCount<KeyType>(tree.treeLeaves());
        EXPECT_EQ(nodeCounts, refCounts);
    }

    focusStart = 0; // slight variation; start from zero instead of 1
    focusEnd   = pad(KeyType(1), 3);
    while (!tree.update(box, codes, focusStart, focusEnd, {})) {}

    {
        TreeNodeIndex lastFocusNode = findNodeAbove(tree.treeLeaves().data(), tree.treeLeaves().size(), focusEnd);
        // tree now focused again on first octant
        EXPECT_TRUE(std::equal(begin(csTree), begin(csTree) + lastFocusNode, tree.treeLeaves().begin()));

        auto nodeCounts = octantNodeCount<KeyType>(tree.treeLeaves());
        EXPECT_EQ(nodeCounts, refCounts);
    }
}

TEST(FocusedOctree, compute)
{
    computeEssentialTree<unsigned>();
    computeEssentialTree<uint64_t>();
}

class MacRefinement : public testing::Test
{
protected:
    using KeyType = uint64_t;
    using T       = double;

    void SetUp() override
    {
        OctreeMaker<KeyType> om;
        om.divide().divide(0);
        for (int j = 0; j < 8; ++j)
            om.divide(0, j);
        om.divide(0, 0, 0);

        // L3 in first octant, L1 otherwise
        leaves = om.makeTree();
        octree.resize(nNodes(leaves));
        ov = octree.data();
        updateInternalTree<KeyType>(leaves, ov);
    }

    std::vector<KeyType> leaves;
    OctreeData<KeyType, CpuTag> octree;
    OctreeView<KeyType> ov;
    std::vector<SourceCenterType<T>> centers;
    std::vector<char> macs;
};

TEST_F(MacRefinement, fullSurface)
{
    Box<T> box(0, 1);
    float invTheta = sqrt(3) / 2 + 1e-6;

    KeyType focusStart = 0;
    KeyType focusEnd   = decodePlaceholderBit(KeyType(011));
    while (!macRefine(octree, leaves, centers, macs, focusEnd, focusEnd, focusStart, focusEnd, invTheta, box)) {}

    int numNodesVertex = 7 + 8;
    int numNodesEdge = 6 + 2 * 8;
    int numNodesFace = 4 + 4 * 8;
    EXPECT_EQ(nNodes(leaves), 64 + 7 + 3 * numNodesFace + 3 * numNodesEdge + numNodesVertex);
}

TEST_F(MacRefinement, noSurface)
{
    Box<T> box(0, 1);
    float invTheta = sqrt(3) / 2 + 1e-6;
    TreeNodeIndex numNodesStart = octree.numLeafNodes;

    KeyType oldFStart  = decodePlaceholderBit(KeyType(0101));
    KeyType oldFEnd    = decodePlaceholderBit(KeyType(011));
    KeyType focusStart = 0;
    KeyType focusEnd   = decodePlaceholderBit(KeyType(011));
    while (!macRefine(octree, leaves, centers, macs, oldFStart, oldFEnd, focusStart, focusEnd, invTheta, box)) {}

    EXPECT_EQ(nNodes(leaves), numNodesStart);
}

TEST_F(MacRefinement, partialSurface)
{
    Box<T> box(0, 1);
    float invTheta = sqrt(3) / 2 + 1e-6;
    TreeNodeIndex numNodesStart = octree.numLeafNodes;

    KeyType oldFStart  = 0;
    KeyType oldFEnd    = decodePlaceholderBit(KeyType(0107));
    KeyType focusStart = 0;
    KeyType focusEnd   = decodePlaceholderBit(KeyType(011));
    while (!macRefine(octree, leaves, centers, macs, oldFStart, oldFEnd, focusStart, focusEnd, invTheta, box)) {}

    EXPECT_EQ(nNodes(leaves), numNodesStart + 5 * 7);
}

} // namespace cstone
