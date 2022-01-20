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
 * @brief  Octree upsweep tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 */

#include "gtest/gtest.h"

#include "cstone/traversal/upsweep.hpp"
#include "cstone/tree/octree_util.hpp"

using namespace cstone;

/*! @brief test upsweep sum with a regular 4x4x4 grid of leaves
 *
 * This creates 64 level-2 leaf nodes. The resulting internal tree should
 * have 9 nodes, the root node and the 8 level-1 nodes.
 */
template<class KeyType>
static void upsweepSum4x4x4()
{
    std::vector<KeyType> cstoneTree = makeUniformNLevelTree<KeyType>(64, 1);

    Octree<KeyType> octree;
    octree.update(cstoneTree.data(), cstoneTree.data() + cstoneTree.size());

    std::vector<unsigned> nodeCounts(octree.numTreeNodes(), 0);
    for (int i = octree.numInternalNodes(); i < octree.numTreeNodes(); ++i)
    {
        nodeCounts[i] = 1;
    }

    auto sumFunction = [](auto a, auto b, auto c, auto d, auto e, auto f, auto g, auto h) {
        return a + b + c + d + e + f + g + h;
    };
    upsweep(octree, nodeCounts.data() + octree.numInternalNodes(), nodeCounts.data(), sumFunction);

    std::vector<unsigned> refNodeCounts{64, 8, 8, 8, 8, 8, 8, 8, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    EXPECT_EQ(nodeCounts, refNodeCounts);
}

TEST(Upsweep, sum4x4x4)
{
    upsweepSum4x4x4<unsigned>();
    upsweepSum4x4x4<uint64_t>();
}

/*! @brief test upsweep sum with an irregular leaf tree
 *
 * The leaf tree is the result of subdividing the root node, then further
 * subdividing octant 0. This results in 15 leaves, so the internal tree
 * should have two nodes: the root and the one internal level-1 node for the
 * first octant. The root points to the one internal node and to leaves [8:15].
 * The internal level-1 nodes points to leaves [0:8].
 */
template<class KeyType>
static void upsweepSumIrregularL2()
{
    std::vector<KeyType> cstoneTree = OctreeMaker<KeyType>{}.divide().divide(0).makeTree();
    Octree<KeyType> octree;
    octree.update(cstoneTree.data(), cstoneTree.data() + cstoneTree.size());

    std::vector<unsigned> nodeCounts(octree.numTreeNodes(), 0);
    for (int i = 0; i < octree.numLeafNodes(); ++i)
    {
        nodeCounts[i + octree.numInternalNodes()] = i;
    }

    auto sumFunction = [](auto a, auto b, auto c, auto d, auto e, auto f, auto g, auto h) {
        return a + b + c + d + e + f + g + h;
    };
    upsweep(octree, nodeCounts.data() + octree.numInternalNodes(), nodeCounts.data(), sumFunction);

    std::vector<unsigned> refNodeCounts{105, 28, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};

    EXPECT_EQ(nodeCounts, refNodeCounts);
}

TEST(Upsweep, sumIrregularL2)
{
    upsweepSumIrregularL2<unsigned>();
    upsweepSumIrregularL2<uint64_t>();
}

/*! @brief an irregular tree with a max depth of 3
 *
 *     - total nodes: 33
 *     - leaf nodes: 29
 *     - internal nodes: 4
 */
template<class KeyType>
static void upsweepSumIrregularL3()
{
    std::vector<KeyType> cstoneTree = OctreeMaker<KeyType>{}.divide().divide(0).divide(0, 2).divide(3).makeTree();
    Octree<KeyType> octree;
    octree.update(cstoneTree.data(), cstoneTree.data() + cstoneTree.size());

    std::vector<unsigned> nodeCounts(octree.numTreeNodes(), 0);
    for (int i = octree.numInternalNodes(); i < octree.numTreeNodes(); ++i)
    {
        nodeCounts[i] = 1;
    }

    auto sumFunction = [](auto a, auto b, auto c, auto d, auto e, auto f, auto g, auto h) {
        return a + b + c + d + e + f + g + h;
    };
    upsweep(octree, nodeCounts.data() + octree.numInternalNodes(), nodeCounts.data(), sumFunction);

    std::vector<unsigned> refNodeCounts{29, 15, 8, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1,  1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    EXPECT_EQ(nodeCounts, refNodeCounts);
}

TEST(Upsweep, sumIrregularL3)
{
    upsweepSumIrregularL3<unsigned>();
    upsweepSumIrregularL3<uint64_t>();
}
