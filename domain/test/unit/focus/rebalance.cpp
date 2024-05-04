/*! @file
 * @brief Test LET rebalancing
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/focus/octree_focus.hpp"
#include "cstone/tree/cs_util.hpp"

namespace cstone
{

template<class KeyType>
static auto computeNodeOps(const OctreeView<KeyType>& octree,
                           const std::vector<unsigned>& leafCounts,
                           const std::vector<char>& csMacs,
                           KeyType focusStart,
                           KeyType focusEnd,
                           unsigned bucketSize,
                           const std::vector<std::tuple<KeyType, int>>& imacs)
{
    std::vector<unsigned> counts(octree.numNodes);

    gsl::span<const TreeNodeIndex> leafToInternal(octree.leafToInternal + octree.numInternalNodes, octree.numLeafNodes);
    gsl::span<const TreeNodeIndex> childOffsets{octree.childOffsets, size_t(octree.numNodes)};
    gsl::span<const TreeNodeIndex> levelRange(octree.levelRange, maxTreeLevel<KeyType>{} + 2);

    scatter(leafToInternal, leafCounts.data(), counts.data());
    upsweep(levelRange, childOffsets, counts.data(), NodeCount<unsigned>{});

    std::vector<char> macs(octree.numNodes);
    scatter(leafToInternal, csMacs.data(), macs.data());

    // transfer values for internal node macs
    for (auto t : imacs)
    {
        auto [key, value] = t;
        TreeNodeIndex idx = locateNode(key, octree.prefixes, octree.levelRange);
        macs[idx]         = value;
    }

    std::vector<int> nodeOps(octree.numNodes);
    rebalanceDecisionEssential({octree.prefixes, size_t(octree.numNodes)}, octree.childOffsets, octree.parents,
                               counts.data(), macs.data(), focusStart, focusEnd, bucketSize, nodeOps.data());
    bool converged =
        protectAncestors(gsl::span<const KeyType>(octree.prefixes, octree.numNodes), octree.parents, nodeOps.data());

    std::vector<int> ret(octree.numLeafNodes);
    gather(leafToInternal, nodeOps.data(), ret.data());

    return std::make_tuple(ret, converged);
}

//! @brief various tests about merge/split decisions based on node counts and MACs
template<class KeyType>
static void rebalanceDecision()
{
    unsigned bucketSize = 1;

    {
        std::vector<KeyType> cstree = OctreeMaker<KeyType>{}.divide().divide(0).divide(7).makeTree();
        OctreeData<KeyType, CpuTag> octree;
        octree.resize(nNodes(cstree));
        updateInternalTree<KeyType>(cstree, octree.data());

        // nodes 14-21 should be fused based on counts, and 14 should be split based on MACs. counts win, nodes are
        // fused
        //                               0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
        std::vector<unsigned> leafCounts{1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
        std::vector<char> macs{/*     */ 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};

        std::vector<int> reference{1, 1, 1, 8, 1, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};

        // internal cell macs in placeholder bit format
        std::vector<std::tuple<KeyType, int>> internalMacs{{1, 1}, {010, 1}, {017, 1}};
        auto [nodeOps, converged] =
            computeNodeOps(octree.data(), leafCounts, macs, cstree[0], cstree[8], bucketSize, internalMacs);

        EXPECT_EQ(nodeOps, reference);
        EXPECT_FALSE(converged);
    }
    {
        std::vector<KeyType> cstree = OctreeMaker<KeyType>{}.divide().divide(0).divide(7).makeTree();
        OctreeData<KeyType, CpuTag> octree;
        octree.resize(nNodes(cstree));
        updateInternalTree<KeyType>(cstree, octree.data());

        // nodes 14-21 should be split/stay based on counts, and should stay based on MACs.
        // MAC wins, nodes stay, but are not split
        //                               0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
        std::vector<unsigned> leafCounts{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 1, 0, 0, 0, 0};
        std::vector<char> macs{/*     */ 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
        //                            ^
        //                            parent of leaf nodes 14-21
        std::vector<int> reference{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

        std::vector<std::tuple<KeyType, int>> internalMacs{{1, 1}, {010, 1}, {017, 1}};
        auto [nodeOps, converged] =
            computeNodeOps(octree.data(), leafCounts, macs, cstree[0], cstree[8], bucketSize, internalMacs);

        EXPECT_EQ(nodeOps, reference);
        EXPECT_TRUE(converged);
    }
    {
        std::vector<KeyType> cstree = OctreeMaker<KeyType>{}.divide().divide(0).divide(7).makeTree();
        OctreeData<KeyType, CpuTag> octree;
        octree.resize(nNodes(cstree));
        updateInternalTree<KeyType>(cstree, octree.data());

        // nodes 14-21 should stay based on counts, and should be fused based on MACs. MAC wins, nodes are fused
        //                               0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
        std::vector<unsigned> leafCounts{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 1, 0, 0, 0, 0};
        std::vector<char> macs{/*     */ 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
        //                            ^
        //                            parent of leaf nodes 14-21
        std::vector<int> reference{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};

        std::vector<std::tuple<KeyType, int>> internalMacs{{1, 1}, {010, 1}, {017, 0}};
        auto [nodeOps, converged] =
            computeNodeOps(octree.data(), leafCounts, macs, cstree[0], cstree[8], bucketSize, internalMacs);

        EXPECT_EQ(nodeOps, reference);
        EXPECT_FALSE(converged);
    }
    {
        // this example has a focus area that cuts through sets of 8 neighboring sibling nodes
        std::vector<KeyType> cstree = OctreeMaker<KeyType>{}.divide().divide(0).divide(1).makeTree();
        OctreeData<KeyType, CpuTag> octree;
        octree.resize(nNodes(cstree));
        updateInternalTree<KeyType>(cstree, octree.data());

        //                               |                     |                       |
        //                               0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
        std::vector<unsigned> leafCounts{1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1};
        std::vector<char> macs{/*     */ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
        //                 root ^  ^  ^
        //   parent of leaves 0-7  |  | parent of leaf nodes 8-15                  ^ here count says split, mac says
        //                                                                           merge, result: stay
        std::vector<int> reference{1, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 1};
        //                                                             ----------------
        //                   these nodes are kept alive because their siblings (8 and 9) are inside the focus and are
        //                   staying

        //                                               root  parent of leaves 0-7
        //                                                  |       |           . parent of leaves 8-15
        std::vector<std::tuple<KeyType, int>> internalMacs{{1, 1}, {010, 1}, {011, 0}};
        auto [nodeOps, converged] =
            computeNodeOps(octree.data(), leafCounts, macs, cstree[2], cstree[10], bucketSize, internalMacs);

        EXPECT_EQ(nodeOps, reference);
        EXPECT_FALSE(converged);
    }
    {
        std::vector<KeyType> cstree = OctreeMaker<KeyType>{}.divide().divide(6).divide(7).makeTree();
        OctreeData<KeyType, CpuTag> octree;
        octree.resize(nNodes(cstree));
        updateInternalTree<KeyType>(cstree, octree.data());

        //                                                                        focus
        //                               |                |                      |-----------------------
        //                               0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
        std::vector<unsigned> leafCounts{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        std::vector<char> macs{/*     */ 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        std::vector<int> reference{1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
        // Check that nodes 6-13, directly adjacent to the focus area can be merged

        std::vector<std::tuple<KeyType, int>> internalMacs{{1, 1}, {016, 0}, {017, 0}};
        auto [nodeOps, converged] =
            computeNodeOps(octree.data(), leafCounts, macs, cstree[14], cstree[22], bucketSize, internalMacs);

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
static void nodeOpsKeepAlive()
{
    {
        std::vector<KeyType> cstree = OctreeMaker<KeyType>{}.divide().divide(0).divide(7).makeTree();
        OctreeData<KeyType, CpuTag> octree;
        octree.resize(nNodes(cstree));
        updateInternalTree<KeyType>(cstree, octree.data());

        // | 0 | 1 2 3 4 5 6 7 8 |
        //       |             |
        //       |             17 18 19 20 21 22 23 24
        //       9 10 11 12 13 14 15 16
        //                                 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
        std::vector<TreeNodeIndex> nodeOps{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        for (size_t i = 0; i < nodeOps.size(); ++i)
        {
            nodeOps[i] = nzAncestorOp(i, octree.prefixes.data(), octree.parents.data(), nodeOps.data());
        }

        //                                   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
        std::vector<TreeNodeIndex> reference{1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        EXPECT_EQ(nodeOps, reference);
    }
}

TEST(FocusedOctree, nodeOpsKeepAlive)
{
    nodeOpsKeepAlive<unsigned>();
    nodeOpsKeepAlive<uint64_t>();
}

TEST(FocusedOctree, keyEnforcement)
{
    using KeyType = unsigned;

    {
        auto cstree = OctreeMaker<KeyType>{}.divide().divide(1).makeTree();
        OctreeData<KeyType, CpuTag> octree_;
        octree_.resize(nNodes(cstree));
        updateInternalTree<KeyType>(cstree, octree_.data());
        auto octree = octree_.data();

        // | 0 | 1 2 3 4 5 6 7 8 |
        //         |
        //         |
        //         9 10 11 12 13 14 15 16
        {
            //                       0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
            std::vector<int> nodeOps{1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            auto status = enforceKeySingle(decodePlaceholderBit(0111u), octree.prefixes, octree.childOffsets,
                                           octree.parents, nodeOps.data());

            std::vector<int> ref{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
            EXPECT_EQ(status, ResolutionStatus::cancelMerge);
            EXPECT_EQ(nodeOps, ref);
        }
        {
            //                       0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
            std::vector<int> nodeOps{1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            auto status = enforceKeySingle(decodePlaceholderBit(01112u), octree.prefixes, octree.childOffsets,
                                           octree.parents, nodeOps.data());

            //                   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
            std::vector<int> ref{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 1, 1};
            EXPECT_EQ(status, ResolutionStatus::rebalance);
            EXPECT_EQ(nodeOps, ref);
        }
        {
            //                       0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
            std::vector<int> nodeOps{1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            auto status = enforceKeySingle(decodePlaceholderBit(0101u), octree.prefixes, octree.childOffsets,
                                           octree.parents, nodeOps.data());

            //                   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
            std::vector<int> ref{1, 8, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
            protectAncestors(gsl::span<const KeyType>(octree.prefixes, octree.numNodes), octree.parents,
                             nodeOps.data());
            EXPECT_EQ(status, ResolutionStatus::rebalance);
            EXPECT_EQ(nodeOps, ref);
        }
        {
            // this tests that the splitting of node 1 does not invalidate the merge of nodes 9-16
            //                       0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
            std::vector<int> nodeOps{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
            //                          ^ injection splits node 1
            auto status = enforceKeySingle(decodePlaceholderBit(0101u), octree.prefixes, octree.childOffsets,
                                           octree.parents, nodeOps.data());
            EXPECT_EQ(status, ResolutionStatus::rebalance);

            status = enforceKeySingle(decodePlaceholderBit(01011u), octree.prefixes, octree.childOffsets,
                                      octree.parents, nodeOps.data());
            EXPECT_EQ(status, ResolutionStatus::failed);

            std::vector<int> ref{1, 8, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
            EXPECT_EQ(nodeOps, ref);
        }
    }
}

} // namespace cstone
