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
 * @brief Test cornerstone octree core functionality
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/tree/octree.hpp"
#include "cstone/tree/octree_util.hpp"

#include "coord_samples/random.hpp"

using namespace cstone;

TEST(CornerstoneOctree, findSearchBounds)
{
    using CodeType = unsigned;

    //                          0   1   2   3   4   5   6   7   8   9
    std::vector<CodeType> codes{3, 10, 11, 14, 16, 16, 16, 18, 19, 21};
    const CodeType* c = codes.data();

    {
        // upward search direction, guess distance from target: 0
        int guess = 3;
        auto probe = findSearchBounds(guess, CodeType(14), c, c + codes.size());
        pair<const CodeType*> reference{c+2, c+4};
        EXPECT_EQ(probe[0]-c, reference[0]-c);
        EXPECT_EQ(probe[1]-c, reference[1]-c);
    }
    {
        // upward search direction, guess distance from target: 1
        int guess = 3;
        auto probe = findSearchBounds(guess, CodeType(15), c, c + codes.size());
        pair<const CodeType*> reference{c+3, c+4};
        EXPECT_EQ(probe[0]-c, reference[0]-c);
        EXPECT_EQ(probe[1]-c, reference[1]-c);
    }
    {
        // upward search direction, guess distance from target: 1
        int guess = 3;
        auto probe = findSearchBounds(guess, CodeType(16), c, c + codes.size());
        pair<const CodeType*> reference{c+3, c+7};
        EXPECT_EQ(probe[0]-c, reference[0]-c);
        EXPECT_EQ(probe[1]-c, reference[1]-c);
    }
    {
        // upward search direction, guess distance from target: 6
        int guess = 0;
        auto probe = findSearchBounds(guess, CodeType(17), c, c + codes.size());
        pair<const CodeType*> reference{c+0, c+8};
        EXPECT_EQ(probe[0]-c, reference[0]-c);
        EXPECT_EQ(probe[1]-c, reference[1]-c);
    }
    {
        // downward search direction
        int guess = 4;
        auto probe = findSearchBounds(guess, CodeType(12), c, c + codes.size());
        pair<const CodeType*> reference{c+2, c+4};
        EXPECT_EQ(probe[0]-c, reference[0]-c);
        EXPECT_EQ(probe[1]-c, reference[1]-c);
    }
    {
        // downward search direction
        int guess = 4;
        auto probe = findSearchBounds(guess, CodeType(11), c, c + codes.size());
        pair<const CodeType*> reference{c+0, c+4};
        EXPECT_EQ(probe[0]-c, reference[0]-c);
        EXPECT_EQ(probe[1]-c, reference[1]-c);
    }
    {
        // downward search direction
        int guess = 4;
        auto probe = findSearchBounds(guess, CodeType(10), c, c + codes.size());
        pair<const CodeType*> reference{c+0, c+4};
        EXPECT_EQ(probe[0]-c, reference[0]-c);
        EXPECT_EQ(probe[1]-c, reference[1]-c);
    }
    {
        // downward search direction
        int guess = 8;
        auto probe = findSearchBounds(guess, CodeType(16), c, c + codes.size());
        pair<const CodeType*> reference{c+0, c+8};
        EXPECT_EQ(probe[0]-c, reference[0]-c);
        EXPECT_EQ(probe[1]-c, reference[1]-c);
    }
    {
        // downward search direction
        int guess = 6;
        auto probe = findSearchBounds(guess, CodeType(16), c, c + codes.size());
        pair<const CodeType*> reference{c+3, c+7};
        EXPECT_EQ(probe[0]-c, reference[0]-c);
        EXPECT_EQ(probe[1]-c, reference[1]-c);
    }
    {
        // direct hit on the last element
        int guess = 9;
        auto probe = findSearchBounds(guess, CodeType(21), c, c + codes.size());
        pair<const CodeType*> reference{c+8, c+10};
        EXPECT_EQ(probe[0]-c, reference[0]-c);
        EXPECT_EQ(probe[1]-c, reference[1]-c);
    }
    {
        // must be able to handle out-of-bounds guess
        int guess = 12;
        auto probe = findSearchBounds(guess, CodeType(16), c, c + codes.size());
        pair<const CodeType*> reference{c+1, c+9};
        EXPECT_EQ(probe[0]-c, reference[0]-c);
        EXPECT_EQ(probe[1]-c, reference[1]-c);
    }

}

//! @brief test that computeNodeCounts correctly counts the number of codes for each node
template<class CodeType>
void checkCountTreeNodes()
{
    std::vector<CodeType> tree = OctreeMaker<CodeType>{}.divide().divide(0).makeTree();

    std::vector<CodeType> codes{tree[1], tree[1], tree[1] + 10, tree[1] + 100, tree[2] - 1,
                                tree[2] + 1,
                                tree[11], tree[11] + 2,
                                tree[12], tree[12] + 1000, tree[12] + 2000, tree[13] - 10,
                                tree[13], tree[13] + 1};

    //  nodeIdx                     0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
    std::vector<unsigned> reference{0, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 2, 0};
    // code start location             0  5  6  6  6  6  6  6  6  6  6  8 12
    // guess start location            0  1  2  3  4  5  6  7  8  9 10 11 12
    // Ntot: 14, nNonZeroNodes: 13 (first and last node are empty), avgNodeCount: 14/13 = 1

    std::vector<unsigned> counts(nNodes(tree));
    computeNodeCounts(tree.data(), counts.data(), nNodes(tree),
                      codes.data(), codes.data() + codes.size(),
                      std::numeric_limits<unsigned>::max());

    EXPECT_EQ(counts, reference);
}

TEST(CornerstoneOctree, countTreeNodes32)
{
    checkCountTreeNodes<unsigned>();
}

TEST(CornerstoneOctree, countTreeNodes64)
{
    checkCountTreeNodes<uint64_t>();
}

template<class CodeType, class LocalIndex>
void rebalanceDecision()
{
    std::vector<CodeType> tree = OctreeMaker<CodeType>{}.divide().divide(0).makeTree();

    unsigned bucketSize = 4;
    std::vector<unsigned> counts{1,1,1,0,0,0,0,0, 2, 3, 4, 5, 6, 7, 8};

    std::vector<LocalIndex> nodeOps(nNodes(tree));
    bool converged = rebalanceDecision(tree.data(), counts.data(), nNodes(tree), bucketSize, nodeOps.data());

    std::vector<LocalIndex> reference{1,0,0,0,0,0,0,0, 1, 1, 1, 8, 8, 8, 8};
    EXPECT_EQ(nodeOps, reference);
    EXPECT_FALSE(converged);
}

TEST(CornerstoneOctree, rebalanceDecision)
{
    rebalanceDecision<unsigned, unsigned>();
    rebalanceDecision<uint64_t, unsigned>();
}

template<class CodeType, class LocalIndex>
void rebalanceDecisionSingleRoot()
{
    std::vector<CodeType> tree = OctreeMaker<CodeType>{}.makeTree();

    unsigned bucketSize = 4;
    std::vector<unsigned> counts{1};

    std::vector<LocalIndex> nodeOps(nNodes(tree));
    bool converged = rebalanceDecision(tree.data(), counts.data(), nNodes(tree), bucketSize, nodeOps.data());

    std::vector<LocalIndex> reference{1};
    EXPECT_EQ(nodeOps, reference);
    EXPECT_TRUE(converged);
}

TEST(CornerstoneOctree, rebalanceDecisionSingleRoot)
{
    rebalanceDecisionSingleRoot<unsigned, unsigned>();
    rebalanceDecisionSingleRoot<uint64_t, unsigned>();
}

/*! @brief test behavior of a maximum-depth tree under rebalancing
 *
 *  Node 0 is at the lowest octree level (10 or 21) and its particle
 *  count is bigger than the bucket size. This test verifies that
 *  this tree stays invariant under rebalancing because the capacity
 *  of the underlying 30-bit or 63 bit Morton code is exhausted.
 */
template<class CodeType>
void rebalanceInsufficentResolution()
{
    constexpr int bucketSize = 1;

    OctreeMaker<CodeType> octreeMaker;
    for (unsigned level = 0; level < maxTreeLevel<CodeType>{}; ++level)
        octreeMaker.divide({}, level);

    std::vector<CodeType> tree = octreeMaker.makeTree();

    std::vector<unsigned> counts(nNodes(tree), 1);
    counts[0] = bucketSize + 1;

    std::vector<TreeNodeIndex> nodeOps(tree.size());
    // the first node has two particles, one more than the bucketSize
    // since the first node is at the maximum subdivision layer, the tree
    // can't be further refined to satisfy the bucketSize
    bool converged = rebalanceDecision(tree.data(), counts.data(), nNodes(tree), bucketSize, nodeOps.data());

    std::vector<TreeNodeIndex> reference(tree.size(), 1);
    reference[nNodes(tree)] = 0; // last value is for the scan result, irrelevant here

    EXPECT_EQ(nodeOps, reference);
    EXPECT_TRUE(converged);
}

TEST(CornerstoneOctree, rebalanceInsufficientResolution)
{
    rebalanceInsufficentResolution<unsigned>();
    rebalanceInsufficentResolution<uint64_t>();
}

//! @brief check that nodes can be fused at the start of the tree
template<class CodeType>
void rebalanceTree()
{
    std::vector<CodeType> tree = OctreeMaker<CodeType>{}.divide().divide(0).makeTree();

    std::vector<TreeNodeIndex> nodeOps{1,0,0,0,0,0,0,0,1,8,1,1,1,1,8,0};
    ASSERT_EQ(nodeOps.size(), tree.size());

    std::vector<CodeType> newTree;
    rebalanceTree(tree, newTree, nodeOps.data());

    std::vector<CodeType> reference = OctreeMaker<CodeType>{}.divide().divide(2).divide(7).makeTree();
    EXPECT_EQ(newTree, reference);
}

TEST(CornerstoneOctree, rebalance)
{
    rebalanceTree<unsigned>();
    rebalanceTree<uint64_t>();
}

template<class I>
void checkOctreeWithCounts(const std::vector<I>& tree, const std::vector<unsigned>& counts, int bucketSize,
                           const std::vector<I>& mortonCodes, bool relaxBucketCount)
{
    using CodeType = I;
    EXPECT_TRUE(checkOctreeInvariants(tree.data(), nNodes(tree)));

    int nParticles = mortonCodes.size();

    // check that referenced particles are within specified range
    for (std::size_t nodeIndex = 0; nodeIndex < nNodes(tree); ++nodeIndex)
    {
        int nodeStart = std::lower_bound(begin(mortonCodes), end(mortonCodes), tree[nodeIndex]) -
                        begin(mortonCodes);

        int nodeEnd = std::lower_bound(begin(mortonCodes), end(mortonCodes), tree[nodeIndex+1]) -
                      begin(mortonCodes);

        // check that counts are correct
        EXPECT_EQ(nodeEnd - nodeStart, counts[nodeIndex]);
        if(!relaxBucketCount)
        {
            EXPECT_LE(counts[nodeIndex], bucketSize);
        }

        if (counts[nodeIndex])
        {
            ASSERT_LT(nodeStart, nParticles);
        }

        for (std::size_t i = nodeStart; i < counts[nodeIndex]; ++i)
        {
            CodeType iCode = mortonCodes[i];
            EXPECT_TRUE(tree[nodeIndex] <= iCode);
            EXPECT_TRUE(iCode < tree[nodeIndex+1]);
        }
    }
}

class ComputeOctreeTester : public testing::TestWithParam<int>
{
public:
    template<class I, template <class...> class CoordinateType>
    void check(int bucketSize)
    {
        using CodeType = I;
        Box<double> box{-1, 1};

        int nParticles = 100000;

        CoordinateType<double, CodeType> randomBox(nParticles, box);
        std::vector<CodeType> codes = randomBox.mortonCodes();

        auto [tree, counts] = computeOctree(codes.data(), codes.data() + nParticles, bucketSize);

        checkOctreeWithCounts(tree, counts, bucketSize, codes, false);

        // update with unchanged particle codes
        updateOctree(codes.data(), codes.data() + nParticles, bucketSize, tree, counts);
        checkOctreeWithCounts(tree, counts, bucketSize, codes, false);

        // range of smallest treeNode
        CodeType minRange = std::numeric_limits<CodeType>::max();
        for (int i = 0; i < nNodes(tree); ++i)
            minRange = std::min(minRange, tree[i+1] - tree[i]);

        // change codes a bit
        std::mt19937 gen(42);
        std::uniform_int_distribution<std::make_signed_t<CodeType>> displace(-minRange, minRange);

        for (auto& code : codes)
            code = std::max(std::make_signed_t<I>(0), std::min(std::make_signed_t<I>(code) + displace(gen),
                                                               std::make_signed_t<I>(nodeRange<I>(0)-1)));

        std::sort(begin(codes), end(codes));
        updateOctree(codes.data(), codes.data() + nParticles, bucketSize, tree, counts);
        // count < bucketSize may not be true anymore, but node counts still have to be correct
        checkOctreeWithCounts(tree, counts, bucketSize, codes, true);
    }
};

TEST_P(ComputeOctreeTester, pingPongRandomNormal32)
{
    check<unsigned, RandomGaussianCoordinates>(GetParam());
}

TEST_P(ComputeOctreeTester, pingPongRandomNormal64)
{
    check<uint64_t, RandomGaussianCoordinates>(GetParam());
}

std::array<int, 3> bucketSizesPP{64, 1024, 10000};

INSTANTIATE_TEST_SUITE_P(RandomBoxPP, ComputeOctreeTester, testing::ValuesIn(bucketSizesPP));


TEST(CornerstoneOctree, computeHaloRadii)
{
    using CodeType = unsigned;

    std::vector<CodeType> tree{0,8,16,24,32};

    std::vector<CodeType> particleCodes{ 0,4, 8,14, 20, 24,25,26,31 };
    std::vector<float>    smoothingLs{   2,1, 4,3,   5, 8,2,1,3};
    std::vector<float>    hMaxPerNode{    4,   8,   10,  16};

    std::vector<int> ordering(particleCodes.size());
    std::iota(begin(ordering), end(ordering), 0);

    std::vector<float> probe(hMaxPerNode.size());

    computeHaloRadii(tree.data(), nNodes(tree), particleCodes.data(), particleCodes.data() + particleCodes.size(),
                     ordering.data(), smoothingLs.data(), probe.data());

    EXPECT_EQ(probe, hMaxPerNode);
}

TEST(CornerstoneOctree, nodeMaxRegression)
{
    std::vector<unsigned> tree{0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 40, 48, 56, 64, 128, 192, 256, 320, 384,
                               448, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 8192, 12288, 16384, 20480, 24576,
                               28672, 32768, 65536, 98304, 131072, 163840, 196608, 229376, 262144, 524288, 786432, 1048576,
                               1310720, 1572864, 1835008, 2097152, 4194304, 6291456, 8388608, 10485760, 12582912, 14680064,
                               16777216, 33554432, 50331648, 67108864, 83886080, 100663296, 117440512, 134217728, 268435456,
                               402653184, 536870912, 671088640, 805306368, 939524096, 956301312, 973078528, 989855744, 1006632960,
                               1023410176, 1040187392, 1056964608, 1059061760, 1061158912, 1063256064, 1065353216, 1067450368,
                               1069547520, 1071644672, 1071906816, 1072168960, 1072431104, 1072693248, 1072955392, 1073217536,
                               1073479680, 1073512448, 1073545216, 1073577984, 1073610752, 1073643520, 1073676288, 1073709056,
                               1073713152, 1073717248, 1073721344, 1073725440, 1073729536, 1073733632, 1073737728, 1073738240,
                               1073738752, 1073739264, 1073739776, 1073740288, 1073740800, 1073741312, 1073741376, 1073741440,
                               1073741504, 1073741568, 1073741632, 1073741696, 1073741760, 1073741768, 1073741776, 1073741784,
                               1073741792, 1073741800, 1073741808, 1073741816, 1073741817, 1073741818, 1073741819, 1073741820,
                               1073741821, 1073741822, 1073741823, 1073741824};

    EXPECT_TRUE(checkOctreeInvariants(tree.data(), nNodes(tree)));

    std::vector<unsigned> nodeCounts(nNodes(tree), 0);
    nodeCounts[0]        = 2;
    *nodeCounts.rbegin() = 2;

    std::vector<unsigned> codes{0, 0, 1073741823, 1073741823};

    {
        std::vector<unsigned> countsProbe(nNodes(tree));
        computeNodeCounts(tree.data(), countsProbe.data(), nNodes(tree), codes.data(), codes.data() + codes.size(),
                          std::numeric_limits<unsigned>::max());
        EXPECT_EQ(nodeCounts, countsProbe);
    }

    std::vector<double> h{0.2, 0.2, 0.2, 0.2};
    std::vector<int> ordering{0,1,2,3};

    std::vector<double> hMaxPerNode(nNodes(tree), 0);
    computeHaloRadii(tree.data(), nNodes(tree), codes.data(), codes.data() + codes.size(), ordering.data(), h.data(),
                     hMaxPerNode.data());

    std::vector<double> refhMaxPerNode(nNodes(tree));
    refhMaxPerNode[0] = 0.4;
    refhMaxPerNode[nNodes(tree)-1] = 0.4;

    EXPECT_EQ(refhMaxPerNode, hMaxPerNode);
}