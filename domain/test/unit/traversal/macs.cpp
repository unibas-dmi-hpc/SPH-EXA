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

#include "cstone/traversal/macs.hpp"
#include "cstone/tree/octree_util.hpp"

namespace cstone
{

TEST(Macs, minDistanceSq)
{
    using KeyType = uint64_t;
    using T = double;
    constexpr size_t maxCoord = 1u << maxTreeLevel<KeyType>{};
    constexpr T unitLengthSq = T(1.) / (maxCoord * maxCoord);

    Box<T> box(0, 2, 0, 3, 0, 4);

    {
        IBox a(0, 1, 0, 1, 0, 1);
        IBox b(2, 3, 0, 1, 0, 1);

        T probe1 = minDistanceSq<KeyType>(a, b, box);
        T probe2 = minDistanceSq<KeyType>(b, a, box);
        T reference = box.lx() * box.lx() * unitLengthSq;

        EXPECT_DOUBLE_EQ(probe1, reference);
        EXPECT_DOUBLE_EQ(probe2, reference);
    }
    {
        IBox a(0, 1, 0, 1, 0, 1);
        IBox b(0, 1, 2, 3, 0, 1);

        T probe1 = minDistanceSq<KeyType>(a, b, box);
        T probe2 = minDistanceSq<KeyType>(b, a, box);
        T reference = box.ly() * box.ly() * unitLengthSq;

        EXPECT_DOUBLE_EQ(probe1, reference);
        EXPECT_DOUBLE_EQ(probe2, reference);
    }
    {
        IBox a(0, 1, 0, 1, 0, 1);
        IBox b(0, 1, 0, 1, 2, 3);

        T probe1 = minDistanceSq<KeyType>(a, b, box);
        T probe2 = minDistanceSq<KeyType>(b, a, box);
        T reference = box.lz() * box.lz() * unitLengthSq;

        EXPECT_DOUBLE_EQ(probe1, reference);
        EXPECT_DOUBLE_EQ(probe2, reference);
    }
    {
        // this tests the implementation for integer overflow on the largest possible input
        IBox a(0, 1, 0, 1, 0, 1);
        IBox b(maxCoord - 1, maxCoord, 0, 1, 0, 1);

        T probe1 = minDistanceSq<KeyType>(a, b, box);
        T probe2 = minDistanceSq<KeyType>(b, a, box);
        T reference = box.lx() * box.lx() * T(maxCoord - 2) * T(maxCoord - 2) * unitLengthSq;

        EXPECT_DOUBLE_EQ(probe1, reference);
        EXPECT_DOUBLE_EQ(probe2, reference);
    }
}

TEST(Macs, minDistanceSqPbc)
{
    using KeyType = uint64_t;
    using T = double;
    constexpr int maxCoord = 1u << maxTreeLevel<KeyType>{};
    constexpr T unitLengthSq = T(1.) / (T(maxCoord) * T(maxCoord));

    {
        Box<T> box(0, 1, 0, 1, 0, 1, true, false, false);
        IBox a(0, 1, 0, 1, 0, 1);
        IBox b(maxCoord - 1, maxCoord, 0, 1, 0, 1);

        T probe1 = minDistanceSq<KeyType>(a, b, box);
        T probe2 = minDistanceSq<KeyType>(b, a, box);

        EXPECT_DOUBLE_EQ(probe1, 0.0);
        EXPECT_DOUBLE_EQ(probe2, 0.0);
    }
    {
        Box<T> box(0, 1, 0, 1, 0, 1, false, true, false);
        IBox a(0, 1);
        IBox b(0, 1, maxCoord - 1, maxCoord, 0, 1);

        T probe1 = minDistanceSq<KeyType>(a, b, box);
        T probe2 = minDistanceSq<KeyType>(b, a, box);

        EXPECT_DOUBLE_EQ(probe1, 0.0);
        EXPECT_DOUBLE_EQ(probe2, 0.0);
    }
    {
        Box<T> box(0, 1, 0, 1, 0, 1, false, false, true);
        IBox a(0, 1);
        IBox b(0, 1, 0, 1, maxCoord - 1, maxCoord);

        T probe1 = minDistanceSq<KeyType>(a, b, box);
        T probe2 = minDistanceSq<KeyType>(b, a, box);

        EXPECT_DOUBLE_EQ(probe1, 0.0);
        EXPECT_DOUBLE_EQ(probe2, 0.0);
    }
    {
        Box<T> box(0, 1, 0, 1, 0, 1, true, true, true);
        IBox a(0, 1);
        IBox b(maxCoord / 2 + 1, maxCoord / 2 + 2);

        T probe1 = minDistanceSq<KeyType>(a, b, box);
        T probe2 = minDistanceSq<KeyType>(b, a, box);
        T reference = 3 * T(maxCoord / 2 - 2) * T(maxCoord / 2 - 2) * box.lx() * box.lx() * unitLengthSq;

        EXPECT_DOUBLE_EQ(probe1, reference);
        EXPECT_DOUBLE_EQ(probe2, reference);
    }
}

TEST(Macs, nodeLengthSq)
{
    IBox ibox(0, 1);
    Box<double> box(0, 1, 0, 2, 0, 3);

    double reference = 1. / 1024 * 3;
    double probe = nodeLength<unsigned>(ibox, box);
    EXPECT_DOUBLE_EQ(reference, probe);
}

TEST(Macs, minDistanceMac)
{
    IBox a(0, 1);
    IBox b(6, 8, 0, 1, 0, 1);
    Box<double> box(0, 1);

    bool probe1 = minDistanceMac<unsigned>(a, b, box, 6.0);
    bool probe2 = minDistanceMac<unsigned>(a, b, box, 6.5);

    EXPECT_TRUE(probe1);
    EXPECT_FALSE(probe2);
}

template<class KeyType>
void markMac()
{
    Box<double> box(0, 1);
    std::vector<KeyType> tree = OctreeMaker<KeyType>{}.divide().divide(0).divide(7).makeTree();

    Octree<KeyType> fullTree;
    fullTree.update(tree.data(), tree.data() + tree.size());

    std::vector<char> markings(fullTree.numTreeNodes(), 0);

    float theta = 0.58;
    KeyType focusStart = tree[0];
    KeyType focusEnd = tree[2];
    markMac(fullTree, box, focusStart, focusEnd, 1. / (theta * theta), markings.data());

    //                         internal | leaves
    //                         0 00 70 | 00 - 07       | 1 - 6     | 70 - 77
    std::vector<char> reference{1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
    EXPECT_EQ(reference, markings);
}

TEST(Macs, markMac)
{
    markMac<unsigned>();
    markMac<uint64_t>();
}

} // namespace cstone