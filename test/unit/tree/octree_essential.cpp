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

#include "cstone/tree/octree_essential.hpp"
#include "cstone/tree/octree_util.hpp"

#include "coord_samples/random.hpp"

namespace cstone
{

TEST(OctreeEssential, minDistanceSq)
{
    using I = uint64_t;
    using T = double;
    constexpr size_t maxCoord = 1u<<maxTreeLevel<I>{};
    constexpr T unitLengthSq  = T(1.) / (maxCoord * maxCoord);

    Box<T> box(0,2,0,3,0,4);

    {
        IBox a(0, 1, 0, 1, 0, 1);
        IBox b(2, 3, 0, 1, 0, 1);

        T probe1 = minDistanceSq<T, I>(a, b, box);
        T probe2 = minDistanceSq<T, I>(b, a, box);
        T reference = box.lx() * box.lx() * unitLengthSq;

        EXPECT_DOUBLE_EQ(probe1, reference);
        EXPECT_DOUBLE_EQ(probe2, reference);
    }
    {
        IBox a(0, 1, 0, 1, 0, 1);
        IBox b(0, 1, 2, 3, 0, 1);

        T probe1 = minDistanceSq<T, I>(a, b, box);
        T probe2 = minDistanceSq<T, I>(b, a, box);
        T reference = box.ly() * box.ly() * unitLengthSq;

        EXPECT_DOUBLE_EQ(probe1, reference);
        EXPECT_DOUBLE_EQ(probe2, reference);
    }
    {
        IBox a(0, 1, 0, 1, 0, 1);
        IBox b(0, 1, 0, 1, 2, 3);

        T probe1 = minDistanceSq<T, I>(a, b, box);
        T probe2 = minDistanceSq<T, I>(b, a, box);
        T reference = box.lz() * box.lz() * unitLengthSq;

        EXPECT_DOUBLE_EQ(probe1, reference);
        EXPECT_DOUBLE_EQ(probe2, reference);
    }
    {
        // this tests the implementation for integer overflow on the largest possible input
        IBox a(0, 1, 0, 1, 0, 1);
        IBox b(maxCoord-1, maxCoord, 0, 1, 0, 1);

        T probe1 = minDistanceSq<T, I>(a, b, box);
        T probe2 = minDistanceSq<T, I>(b, a, box);
        T reference = box.lx() * box.lx() * T(maxCoord-2) * T(maxCoord-2) * unitLengthSq;

        EXPECT_DOUBLE_EQ(probe1, reference);
        EXPECT_DOUBLE_EQ(probe2, reference);
    }
}

TEST(OctreeEssential, nodeLengthSq)
{
    IBox ibox(0,1);
    Box<double> box(0,1,0,2,0,3);

    double reference = 1./1024 * 3;
    double probe     = nodeLength<double, unsigned>(ibox, box);
    EXPECT_DOUBLE_EQ(reference, probe);
}


TEST(OctreeEssential, minDistanceMac)
{
    IBox a(0,1);
    IBox b(6,8,0,1,0,1);
    Box<double> box(0,1);

    bool probe1 = minDistanceMac<double, unsigned>(a, b, box, 6.0);
    bool probe2 = minDistanceMac<double, unsigned>(a, b, box, 6.5);

    EXPECT_TRUE(probe1);
    EXPECT_FALSE(probe2);
}

template<class I>
void markMac()
{
    Box<double> box(0,1);
    std::vector<I> tree = OctreeMaker<I>{}.divide().divide(0).divide(7).makeTree();

    Octree<I> fullTree;
    fullTree.update(tree.data(), tree.data() + tree.size());

    std::vector<char> markings(fullTree.nTreeNodes(), 0);

    float theta = 0.58;
    markMac(fullTree, box, 0, 2, 1./(theta*theta), markings.data());

    // first two leaf nodes are in the target range, we don't check any criterion there
    markings[fullTree.toInternal(0)] = 0;
    markings[fullTree.toInternal(1)] = 0;

    //for (int i = 0; i < fullTree.nTreeNodes(); ++i)
    //    std::cout << std::dec << i << " " << std::oct << fullTree.codeStart(i) << " " << int(markings[i]) << std::endl;

    std::vector<char> reference{1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0};
    EXPECT_EQ(reference, markings);
}

TEST(OctreeEssential, markMac)
{
    markMac<unsigned>();
    markMac<uint64_t>();
}

TEST(OctreeEssential, findFringe)
{
    using I = unsigned;
    std::vector<I> cstree = OctreeMaker<I>{}.divide().divide(0).divide(1).makeTree();

    EXPECT_EQ(0, findLowerFringe(0, cstree.data()));
    EXPECT_EQ(0, findLowerFringe(1, cstree.data()));
    EXPECT_EQ(0, findLowerFringe(2, cstree.data()));

    EXPECT_EQ(17, findLowerFringe(17, cstree.data()));

    EXPECT_EQ(8, findUpperFringe(7, cstree.data()));
    EXPECT_EQ(8, findUpperFringe(8, cstree.data()));
    EXPECT_EQ(16, findUpperFringe(9, cstree.data()));
    EXPECT_EQ(16, findUpperFringe(10, cstree.data()));
    EXPECT_EQ(16, findUpperFringe(16, cstree.data()));
    EXPECT_EQ(17, findUpperFringe(17, cstree.data()));
}

//! @brief various tests about merge/split decisions based on node counts and MACs
template<class I>
void rebalanceDecision()
{
    std::vector<I> cstree = OctreeMaker<I>{}.divide().divide(0).divide(7).makeTree();

    Octree<I> tree;
    tree.update(cstree.data(), cstree.data() + cstree.size());

    //for (int i = 0; i < tree.nTreeNodes(); ++i)
    //    std::cout << std::dec << i << " " << std::oct << tree.codeStart(i) << std::endl;

    unsigned bucketSize = 1;

    {
        // nodes 14-21 should be fused based on counts, and 14 should be split based on MACs. counts win, nodes are fused
        //                               0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
        std::vector<unsigned> leafCounts{1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
        std::vector<char>     macs{1,1,1,0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};

        std::vector<int>       reference{1, 1, 1, 8, 1, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};

        std::vector<int> nodeOps(nNodes(cstree));
        bool converged = rebalanceDecisionEssential(tree.cstoneTree().data(), tree.nInternalNodes(), tree.nLeafNodes(), tree.leafParents(),
                                                    leafCounts.data(), macs.data(), 0, 8, bucketSize, nodeOps.data());

        EXPECT_EQ(nodeOps, reference);
        EXPECT_FALSE(converged);
    }
    {
        // nodes 14-21 should be split/stay based on counts, and should stay based on MACs.
        // MAC wins, nodes stay, but are not split
        //                               0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
        std::vector<unsigned> leafCounts{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 1, 0, 0, 0, 0};
        std::vector<char>     macs{1,1,1,0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
        //                             ^
        //                             parent of leaf nodes 14-21
        std::vector<int>       reference{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

        std::vector<int> nodeOps(nNodes(cstree));
        bool converged = rebalanceDecisionEssential(tree.cstoneTree().data(), tree.nInternalNodes(), tree.nLeafNodes(), tree.leafParents(),
                                                    leafCounts.data(), macs.data(), 0, 8, bucketSize, nodeOps.data());

        EXPECT_EQ(nodeOps, reference);
        EXPECT_TRUE(converged);
    }
    {
        // nodes 14-21 should stay based on counts, and should be fused based on MACs. MAC wins, nodes are fused
        EXPECT_EQ(tree.parent(tree.toInternal(14)), 2);
        //                               0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
        std::vector<unsigned> leafCounts{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 1, 0, 0, 0, 0};
        std::vector<char>     macs{1,1,0,0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
        //                             ^
        //                             parent of leaf nodes 14-21
        std::vector<int>       reference{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};

        std::vector<int> nodeOps(nNodes(cstree));
        bool converged = rebalanceDecisionEssential(tree.cstoneTree().data(), tree.nInternalNodes(), tree.nLeafNodes(), tree.leafParents(),
                                                    leafCounts.data(), macs.data(), 0, 8, bucketSize, nodeOps.data());

        EXPECT_EQ(nodeOps, reference);
        EXPECT_FALSE(converged);
    }
    {
        // this example has a focus area that cuts through sets of 8 neighboring sibling nodes
        std::vector<I> cstree = OctreeMaker<I>{}.divide().divide(0).divide(1).makeTree();

        Octree<I> tree;
        tree.update(cstree.data(), cstree.data() + cstree.size());
        // nodes 14-21 should stay based on counts, and should be fused based on MACs. MAC wins, nodes are fused
        EXPECT_EQ(tree.parent(tree.toInternal(14)), 2);
        //                               |                    |  |                    |
        //                               0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
        std::vector<unsigned> leafCounts{1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 2, 1, 1, 1, 1, 1};
        std::vector<char>  macs{1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
        //                 root ^  ^  ^
        //   parent of leaves 0-7  |  | parent of leaf nodes 8-15
        std::vector<int>       reference{1, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 1};
        //                                                             ----------------
        //                   these nodes are kept alive because their siblings (8 and 9) are inside the focus and are staying
        std::vector<int> nodeOps(nNodes(cstree));
        bool converged = rebalanceDecisionEssential(tree.cstoneTree().data(), tree.nInternalNodes(), tree.nLeafNodes(), tree.leafParents(),
                                                    leafCounts.data(), macs.data(), 2, 10, bucketSize, nodeOps.data());

        EXPECT_EQ(nodeOps, reference);
        EXPECT_FALSE(converged);
    }
}

TEST(OctreeEssential, rebalanceDecision)
{
    rebalanceDecision<unsigned>();
    rebalanceDecision<uint64_t>();
}

template<class I>
void printNodes(const Octree<I>& tree)
{
    auto csFocus = tree.cstoneTree();

    TreeNodeIndex octant1 = std::lower_bound(begin(csFocus), end(csFocus), pad(I(1), 3)) - begin(csFocus);
    TreeNodeIndex octant2 = std::lower_bound(begin(csFocus), end(csFocus), pad(I(2), 3)) - begin(csFocus);
    TreeNodeIndex octant3 = std::lower_bound(begin(csFocus), end(csFocus), pad(I(3), 3)) - begin(csFocus);
    TreeNodeIndex octant4 = std::lower_bound(begin(csFocus), end(csFocus), pad(I(4), 3)) - begin(csFocus);
    TreeNodeIndex octant7 = std::lower_bound(begin(csFocus), end(csFocus), pad(I(7), 3)) - begin(csFocus);
    //for (int i = octant7; i < nNodes(csFocus); ++i)
    //    std::cout << std::oct << csFocus[i] << "\t" << counts[i] << std::endl;
    std::cout << "total: " << tree.nLeafNodes() << std::endl;
    std::cout << "octant 1-2 " << octant2 - octant1 << std::endl;
    std::cout << "octant 3-4 " << octant4 - octant3 << std::endl;
    std::cout << "octant 7-end " << nNodes(csFocus) - octant7 << std::endl;
}

template<class I>
void computeEssentialTree()
{
    Box<double> box{-1, 1};
    int nParticles = 100000;
    unsigned csBucketSize = 16;

    RandomCoordinates<double, I> randomBox(nParticles, box);
    std::vector<I> codes = randomBox.mortonCodes();

    auto [csTree, csCounts] = computeOctree(codes.data(), codes.data() + nParticles, csBucketSize);
    std::cout << "nNodes(csTree): " << nNodes(csTree) << std::endl;

    unsigned bucketSize = 16;
    float theta         = 1.0;

    I focusStart = 0;
    I focusEnd   = pad(I(1), 3);
    auto [tree, counts] = computeOctreeEssential(codes.data(), codes.data() + nParticles, focusStart, focusEnd, bucketSize,
                                                 theta, box, csTree);
    std::cout << std::dec << "nNodes(csTree): " << nNodes(csTree) << std::endl;

    printNodes(tree);
    focusStart = pad(I(6), 3);
    focusEnd  = pad(I(7), 3);

    bool converged = false;
    while (!converged)
    {
        converged = updateOctreeEssential(codes.data(), codes.data() + nParticles, focusStart, focusEnd, bucketSize,
                                          theta, box, csTree, tree, counts);
    }

    printNodes(tree);
    focusStart = 0;
    focusEnd   = pad(I(1), 3);

    converged = false;
    while (!converged)
    {
        converged = updateOctreeEssential(codes.data(), codes.data() + nParticles, focusStart, focusEnd, bucketSize,
                                          theta, box, csTree, tree, counts);
    }

    printNodes(tree);

    EXPECT_GT(tree.nTreeNodes(), 0);
}

TEST(OctreeEssential, compute)
{
    computeEssentialTree<unsigned>();
}

} // namespace cstone