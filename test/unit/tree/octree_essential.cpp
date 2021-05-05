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
        // nodes 14-21 should be fused based on counts, and should stay based on MACs. counts win, nodes are fused
        int converged = 0;
        //                               0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
        std::vector<unsigned> leafCounts{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
        std::vector<char>     macs{1,1,1,0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};

        std::vector<int>       reference{1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 8, 8, 8, 8, 1, 0, 0, 0, 0, 0, 0, 0};

        std::vector<int> nodeOps(nNodes(cstree));
        rebalanceDecisionEssential(tree.cstoneTree().data(), tree.nInternalNodes(), tree.nLeafNodes(), tree.leafParents(),
                                   leafCounts.data(), macs.data(), 0, 8, bucketSize, nodeOps.data(), &converged);

        EXPECT_EQ(nodeOps, reference);
        EXPECT_GT(converged, 0);
    }
    {
        // nodes 14-21 should stay based on counts, and should stay based on MACs. nodes stay
        int converged = 0;
        //                               0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
        std::vector<unsigned> leafCounts{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0};
        std::vector<char>     macs{1,1,1,0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
        //                             ^
        //                             parent of leaf nodes 14-21
        std::vector<int>       reference{1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 8, 8, 8, 8, 1, 1, 1, 1, 1, 1, 1, 1};

        std::vector<int> nodeOps(nNodes(cstree));
        rebalanceDecisionEssential(tree.cstoneTree().data(), tree.nInternalNodes(), tree.nLeafNodes(), tree.leafParents(),
                                   leafCounts.data(), macs.data(), 0, 8, bucketSize, nodeOps.data(), &converged);

        EXPECT_EQ(nodeOps, reference);
        EXPECT_GT(converged, 0);
    }
    {
        // nodes 14-21 should stay based on counts, and should be fused based on MACs. MAC wins, nodes are fused
        EXPECT_EQ(tree.parent(tree.toInternal(14)), 2);
        int converged = 0;
        //                               0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
        std::vector<unsigned> leafCounts{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0};
        std::vector<char>     macs{1,1,0,0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
        //                             ^
        //                             parent of leaf nodes 14-21
        std::vector<int>       reference{1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 8, 8, 8, 8, 1, 0, 0, 0, 0, 0, 0, 0};

        std::vector<int> nodeOps(nNodes(cstree));
        rebalanceDecisionEssential(tree.cstoneTree().data(), tree.nInternalNodes(), tree.nLeafNodes(), tree.leafParents(),
                                   leafCounts.data(), macs.data(), 0, 8, bucketSize, nodeOps.data(), &converged);

        EXPECT_EQ(nodeOps, reference);
        EXPECT_GT(converged, 0);
    }
}

TEST(OctreeEssential, rebalanceDecision)
{
    rebalanceDecision<unsigned>();
    rebalanceDecision<uint64_t>();
}

template<class I>
void computeEssentialTree()
{
    Box<double> box{-1, 1};

    int nParticles = 100000;

    RandomCoordinates<double, I> randomBox(nParticles, box);
    std::vector<I> codes = randomBox.mortonCodes();
}

} // namespace cstone