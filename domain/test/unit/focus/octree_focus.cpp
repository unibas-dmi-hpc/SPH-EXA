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
#include "cstone/tree/octree_util.hpp"

#include "coord_samples/random.hpp"

namespace cstone
{

template<class KeyType>
static auto computeNodeOps(const Octree<KeyType>& octree,
                           const std::vector<unsigned>& leafCounts,
                           const std::vector<char>& csMacs,
                           KeyType focusStart,
                           KeyType focusEnd,
                           unsigned bucketSize,
                           const std::vector<std::tuple<int, int, int>>& imacs)
{
    std::vector<unsigned> counts(octree.numTreeNodes());
    upsweepSum<unsigned>(octree, leafCounts, counts);

    std::vector<char> macs(octree.numTreeNodes());
    upsweepSum<char>(octree, gsl::span<const char>(csMacs.data() + octree.numInternalNodes(), octree.numLeafNodes()),
                     macs);

    // transfer values for internal node macs
    for (auto t : imacs)
    {
        auto [keyStub, level, value] = t;

        KeyType keyStart  = pad(KeyType(keyStub), 3*level);
        TreeNodeIndex idx = octree.locate(keyStart, keyStart + nodeRange<KeyType>(level));
        macs[idx] = value;
    }

    std::vector<int> nodeOps(octree.numTreeNodes());
    bool converged = rebalanceDecisionEssential(octree.nodeKeys(), octree.childOffsets(), octree.parents(),
                                                counts.data(), macs.data(), octree.numTreeNodes(), focusStart, focusEnd,
                                                bucketSize, nodeOps.data());

    std::vector<int> ret(octree.numLeafNodes());
    octree.template extractLeaves<int>(nodeOps, ret);

    return std::make_tuple(ret, converged);
}

//! @brief various tests about merge/split decisions based on node counts and MACs
template<class KeyType>
static void rebalanceDecision()
{
    std::vector<KeyType> cstree = OctreeMaker<KeyType>{}.divide().divide(0).divide(7).makeTree();
    Octree<KeyType> tdtree;
    tdtree.update(cstree.data(), nNodes(cstree));

    unsigned bucketSize = 1;

    {
        // nodes 14-21 should be fused based on counts, and 14 should be split based on MACs. counts win, nodes are
        // fused
        //                               0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
        std::vector<unsigned> leafCounts{1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
        std::vector<char>  macs{1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};

        std::vector<int>       reference{1, 1, 1, 8, 1, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};

        // (keyStub, level, value) tuples for internal cell macs
        std::vector<std::tuple<int, int, int>> internalMacs{{0, 0, 1}, {0, 1, 1}, {7, 1, 1}};
        auto [nodeOps, converged] =
            computeNodeOps(tdtree, leafCounts, macs, cstree[0], cstree[8], bucketSize, internalMacs);

        EXPECT_EQ(nodeOps, reference);
        EXPECT_FALSE(converged);
    }
    {
        // nodes 14-21 should be split/stay based on counts, and should stay based on MACs.
        // MAC wins, nodes stay, but are not split
        //                               0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
        std::vector<unsigned> leafCounts{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 1, 0, 0, 0, 0};
        std::vector<char>  macs{1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
        //                            ^
        //                            parent of leaf nodes 14-21
        std::vector<int>       reference{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

        std::vector<std::tuple<int, int, int>> internalMacs{{0, 0, 1}, {0, 1, 1}, {7, 1, 1}};
        auto [nodeOps, converged] =
            computeNodeOps(tdtree, leafCounts, macs, cstree[0], cstree[8], bucketSize, internalMacs);

        EXPECT_EQ(nodeOps, reference);
        EXPECT_TRUE(converged);
    }
    {
        // nodes 14-21 should stay based on counts, and should be fused based on MACs. MAC wins, nodes are fused
        //                               0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
        std::vector<unsigned> leafCounts{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 1, 0, 0, 0, 0};
        std::vector<char>  macs{1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
        //                            ^
        //                            parent of leaf nodes 14-21
        std::vector<int>       reference{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};

        std::vector<std::tuple<int, int, int>> internalMacs{{0, 0, 1}, {0, 1, 1}, {7, 1, 0}};
        auto [nodeOps, converged] =
            computeNodeOps(tdtree, leafCounts, macs, cstree[0], cstree[8], bucketSize, internalMacs);

        EXPECT_EQ(nodeOps, reference);
        EXPECT_FALSE(converged);
    }
    {
        // this example has a focus area that cuts through sets of 8 neighboring sibling nodes
        std::vector<KeyType> cstree = OctreeMaker<KeyType>{}.divide().divide(0).divide(1).makeTree();

        Octree<KeyType> octree;
        octree.update(cstree.data(), nNodes(cstree));
        //                               |                     |                       |
        //                               0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
        std::vector<unsigned> leafCounts{1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1};
        std::vector<char>  macs{1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
        //                 root ^  ^  ^
        //   parent of leaves 0-7  |  | parent of leaf nodes 8-15                  ^ here count says split, mac says
        //                                                                           merge, result: stay
        std::vector<int>       reference{1, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 1};
        //                                                             ----------------
        //                   these nodes are kept alive because their siblings (8 and 9) are inside the focus and are
        //                   staying

        std::vector<std::tuple<int, int, int>> internalMacs{{0, 0, 1}, {0, 1, 1}, {1, 1, 0}};
        auto [nodeOps, converged] =
            computeNodeOps(octree, leafCounts, macs, cstree[2], cstree[10], bucketSize, internalMacs);

        EXPECT_EQ(nodeOps, reference);
        EXPECT_FALSE(converged);
    }
    {
        std::vector<KeyType> cstree = OctreeMaker<KeyType>{}.divide().divide(6).divide(7).makeTree();

        Octree<KeyType> octree;
        octree.update(cstree.data(), nNodes(cstree));

        //                                                                        focus
        //                               |                |                      |-----------------------
        //                               0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
        std::vector<unsigned> leafCounts{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        std::vector<char>  macs{1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        std::vector<int>       reference{1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
        // Check that nodes 6-13, directly adjacent to the focus area can be merged

        std::vector<std::tuple<int, int, int>> internalMacs{{0, 0, 1}, {6, 1, 0}, {7, 1, 0}};
        auto [nodeOps, converged] =
            computeNodeOps(octree, leafCounts, macs, cstree[14], cstree[22], bucketSize, internalMacs);

        EXPECT_EQ(nodeOps, reference);
        EXPECT_FALSE(converged);
    }
}

TEST(FocusedOctree, rebalanceDecision)
{
    rebalanceDecision<unsigned>();
    rebalanceDecision<uint64_t>();
}

template<class KeyType>
static void enforceKeys()
{
    auto tree = OctreeMaker<KeyType>{}.divide().divide(1).makeTree();

    {
        std::vector<int> nodeOps(nNodes(tree), 1);
        std::vector<KeyType> injectKeys{pad(KeyType(024), 6)};

        auto status = enforceKeys<KeyType>(tree, injectKeys, nodeOps);
        EXPECT_EQ(status, ResolutionStatus::rebalance);
        EXPECT_EQ(nodeOps[9], 8);
    }
    {
        std::vector<int> nodeOps{1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
        std::vector<KeyType> injectKeys{pad(KeyType(014), 6)};

        auto status = enforceKeys<KeyType>(tree, injectKeys, nodeOps);

        std::vector<int> reference(nNodes(tree), 1);
        EXPECT_EQ(status, ResolutionStatus::cancelMerge);
        EXPECT_EQ(nodeOps, reference);
    }
    {
        std::vector<int> nodeOps{1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
        std::vector<KeyType> injectKeys{pad(KeyType(01), 3)};

        auto status = enforceKeys<KeyType>(tree, injectKeys, nodeOps);

        std::vector<int> reference{1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
        EXPECT_EQ(status, ResolutionStatus::converged);
        EXPECT_EQ(nodeOps, reference);
    }
    {
        std::vector<int> nodeOps{1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
        std::vector<KeyType> injectKeys{pad(KeyType(0101), 9)};

        auto status = enforceKeys<KeyType>(tree, injectKeys, nodeOps);

        std::vector<int> reference(nNodes(tree), 1);
        reference[1] = 8;
        EXPECT_EQ(status, ResolutionStatus::rebalance);
        EXPECT_EQ(nodeOps, reference);
    }
    {
        std::vector<int> nodeOps(nNodes(tree), 1);
        std::vector<KeyType> injectKeys{pad(KeyType(014), 6)};

        auto status = enforceKeys<KeyType>(tree, injectKeys, nodeOps);

        std::vector<int> reference(nNodes(tree), 1);
        EXPECT_EQ(status, ResolutionStatus::converged);
        EXPECT_EQ(nodeOps, reference);
    }
    {
        std::vector<int> nodeOps(nNodes(tree), 1);
        std::vector<KeyType> injectKeys{pad(KeyType(0141), 9)};

        auto status = enforceKeys<KeyType>(tree, injectKeys, nodeOps);

        std::vector<int> reference(nNodes(tree), 1);
        reference[5] = 8;
        EXPECT_EQ(status, ResolutionStatus::rebalance);
        EXPECT_EQ(nodeOps, reference);
    }
    {
        // this tests that the splitting of node i does not invalidate the merge of node i+1
        std::vector<int> nodeOps{1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
        //                       ^ injection splits the first node
        std::vector<KeyType> injectKeys{pad(KeyType(01), 6)}; // 010000000000

        auto status = enforceKeys<KeyType>(tree, injectKeys, nodeOps);

        std::vector<int> reference{8, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
        EXPECT_EQ(status, ResolutionStatus::rebalance);
        EXPECT_EQ(nodeOps, reference);
    }
    // two injections affection the same node index
    {
        std::vector<int> nodeOps(nNodes(tree), 1);
        std::vector<KeyType> injectKeys{pad(KeyType(0241), 9), pad(KeyType(024), 6)};

        auto status = enforceKeys<KeyType>(tree, injectKeys, nodeOps);

        EXPECT_EQ(status, ResolutionStatus::failed);
        EXPECT_EQ(nodeOps[9], 8);
    }
    {
        tree = makeRootNodeTree<KeyType>();
        std::vector<int> nodeOps(nNodes(tree), 1);
        std::vector<KeyType> injectKeys{1};

        auto status = enforceKeys<KeyType>(tree, injectKeys, nodeOps);

        EXPECT_EQ(status, ResolutionStatus::failed);
        EXPECT_EQ(nodeOps[0], 8);
    }
    {
        tree = OctreeMaker<KeyType>{}.divide().divide(0).makeTree();
        std::vector<int> nodeOps{1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1};
        std::vector<KeyType> injectKeys{1};

        auto status = enforceKeys<KeyType>(tree, injectKeys, nodeOps);

        std::vector<int> reference(nNodes(tree), 1);
        reference[0] = 8;
        EXPECT_EQ(status, ResolutionStatus::failed);
        EXPECT_EQ(nodeOps, reference);
    }
}

TEST(FocusedOctree, enforceKeys)
{
    enforceKeys<unsigned>();
    enforceKeys<uint64_t>();
}

template<class KeyType>
TreeNodeIndex numNodesInRange(gsl::span<const KeyType> tree, KeyType a, KeyType b)
{
    return std::lower_bound(tree.begin(), tree.end(), b) - std::lower_bound(tree.begin(), tree.end(), a);
}

template<class KeyType>
std::vector<TreeNodeIndex> octantNodeCount(gsl::span<const KeyType> tree)
{
    std::vector<TreeNodeIndex> counts;
    for (int octant = 0; octant < 8; ++octant)
    {
        KeyType range = nodeRange<KeyType>(1);
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
    int nParticles = 200000;
    unsigned csBucketSize = 16;

    auto codes = makeRandomUniformKeys<KeyType>(nParticles);

    auto [csTree, csCounts] = computeOctree(codes.data(), codes.data() + nParticles, csBucketSize);
    Octree<KeyType> globalTree;
    globalTree.update(csTree.data(), nNodes(csTree));

    unsigned bucketSize = 16;
    float theta = 1.0;
    FocusedOctreeSingleNode<KeyType> tree(bucketSize, theta);

    // sorted reference tree node counts in each (except focus) octant at the 1st division level
    std::vector<TreeNodeIndex> refCounts{92, 302, 302, 302, 1184, 1184, 1184, /*4131*/};

    KeyType focusStart = 1;
    KeyType focusEnd = pad(KeyType(1), 3);

    tree.update(box, codes, focusStart, focusEnd, {});
    // The focus boundaries have to be contained in the tree, even after just one update step.
    // This example here is the worst-case scenario with a focus boundary at the highest possible
    // octree subdivision level. Key-0 is always present, so the node with Key-1 is always at index 1, if present.
    EXPECT_EQ(tree.treeLeaves()[1], focusStart);

    // update until converged
    while (!tree.update(box, codes, focusStart, focusEnd, {})) {}

    {
        // the first node in the cornerstone tree that starts at or above focusStart
        TreeNodeIndex firstCstoneNode   = findNodeAbove<KeyType>(csTree, focusStart);
        TreeNodeIndex matchingFocusNode = findNodeAbove(tree.treeLeaves(), csTree[firstCstoneNode]);
        // in the focus area (the first octant) the essential tree and the csTree are identical
        TreeNodeIndex lastFocusNode = findNodeAbove(tree.treeLeaves(), focusEnd);

        EXPECT_TRUE(matchingFocusNode >= 1 && matchingFocusNode < lastFocusNode);

        // We have: focusTree[matchingFocusNode] == csTree[firstCstoneNode]
        // therefore all nodes past matchingFocusNode until lastFocusNode should match those in the csTree
        EXPECT_TRUE(std::equal(tree.treeLeaves().begin() + matchingFocusNode, tree.treeLeaves().begin() + lastFocusNode,
                               begin(csTree) + firstCstoneNode));

        // From 0 to matchingFocusNode, the focusTree should be identical to the spanningTree
        std::vector<KeyType> spanningKeys{0, focusStart, nodeRange<KeyType>(0)};
        auto spanningTree = computeSpanningTree<KeyType>(spanningKeys);
        EXPECT_EQ(tree.treeLeaves().first(matchingFocusNode), gsl::span<KeyType>(spanningTree.data(), matchingFocusNode));

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
        TreeNodeIndex lastFocusNode = findNodeAbove(tree.treeLeaves(), focusEnd);
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

} // namespace cstone