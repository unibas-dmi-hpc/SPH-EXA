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
 * @brief Test multipole acceptance criteria
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/focus/source_center.hpp"
#include "cstone/traversal/macs.hpp"
#include "cstone/tree/octree_util.hpp"

#include "coord_samples/random.hpp"

namespace cstone
{

TEST(Macs, minPointDistance)
{
    using T = double;
    using KeyType = unsigned;

    constexpr unsigned mc = maxCoord<KeyType>{};

    {
        Box<T> box(0, 1);
        IBox ibox(0, mc / 2);

        T px = (mc/2.0 + 1) / mc;
        Vec3<T> X{px, px, px};

        auto [center, size] = centerAndSize<KeyType>(ibox, box);

        T probe = std::sqrt(norm2(minDistance(X, center, size, box)));
        EXPECT_NEAR(std::sqrt(3)/mc, probe, 1e-10);
    }
}

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

TEST(Macs, minMac)
{
    using T = double;

    {
        Vec3<T> cA{0.5, 0.5, 0.5};
        Vec3<T> sA{0.5, 0.5, 0.5};

        Vec3<T> cB{3.5, 3.5, 3.5};
        Vec3<T> sB{0.5, 0.5, 0.5};

        EXPECT_TRUE(minMac(cA, sA, cB, sB, Box<T>(0, 4, false), 1.0 / 0.29));
        EXPECT_FALSE(minMac(cA, sA, cB, sB, Box<T>(0, 4, false), 1.0 / 0.28));

        EXPECT_FALSE(minMac(cA, sA, cB, sB, Box<T>(0, 4, true), 1.0));
    }
    {
        Vec3<T> cA{0.5, 0.5, 0.5};
        Vec3<T> sA{1.0, 1.0, 1.0};

        Vec3<T> cB{3.5, 3.5, 3.5};
        Vec3<T> sB{0.5, 0.5, 0.5};

        EXPECT_TRUE(minMac(cA, sA, cB, sB, Box<T>(0, 4, false), 1.0 / 0.39));
        EXPECT_FALSE(minMac(cA, sA, cB, sB, Box<T>(0, 4, false), 1.0 / 0.38));

        EXPECT_FALSE(minMac(cA, sA, cB, sB, Box<T>(0, 4, true), 1.0));
    }
    {
        Vec3<T> cA{0.5, 0.5, 0.5};
        Vec3<T> sA{0.5, 0.5, 0.5};

        Vec3<T> cB{3.5, 3.5, 3.5};
        Vec3<T> sB{1.0, 1.0, 1.0};

        EXPECT_TRUE(minMac(cA, sA, cB, sB, Box<T>(0, 4, false), 1.0 / 0.78));
        EXPECT_FALSE(minMac(cA, sA, cB, sB, Box<T>(0, 4, false), 1.0 / 0.76));

        EXPECT_FALSE(minMac(cA, sA, cB, sB, Box<T>(0, 4, true), 1.0));
    }
}

TEST(Macs, vectorMacAndPbc)
{
    using T = double;

    Box<T> box(0, 1, true);

    Vec3<T> tcenter{0.1, 0.1, 0.1};
    Vec3<T> tsize{0.01, 0.01, 0.01};

    // R = sqrt(0.03) = 0.173
    T mac = 0.03;
    {
        Vec3<T> scenter{0.2, 0.2, 0.2};
        EXPECT_TRUE(vectorMac(scenter, mac, tcenter, tsize));
        EXPECT_TRUE(vectorMacPbc(scenter, mac, tcenter, tsize, box));
    }
    {
        Vec3<T> scenter{0.2101, 0.2101, 0.2101};
        EXPECT_FALSE(vectorMac(scenter, mac, tcenter, tsize));
        EXPECT_FALSE(vectorMacPbc(scenter, mac, tcenter, tsize, box));
    }
    {
        Vec3<T> scenter{1.0, 1.0, 1.0};
        EXPECT_TRUE(vectorMacPbc(scenter, mac, tcenter, tsize, box));
    }
    {
        Vec3<T> scenter{0.9899, 0.9899, 0.9899};
        EXPECT_FALSE(vectorMacPbc(scenter, mac, tcenter, tsize, box));
    }
}

TEST(Macs, minMacMutual)
{
    using T = double;

    Vec3<T> cA{0.5, 0.5, 0.5};
    Vec3<T> sA{0.5, 0.5, 0.5};

    Vec3<T> cB{3.5, 3.5, 3.5};
    Vec3<T> sB{0.5, 0.5, 0.5};

    EXPECT_TRUE(minMacMutual(cA, sA, cB, sB, Box<T>(0, 4, false), 1.0 / 0.29));
    EXPECT_FALSE(minMacMutual(cA, sA, cB, sB, Box<T>(0, 4, false), 1.0 / 0.28));

    EXPECT_FALSE(minMacMutual(cA, sA, cB, sB, Box<T>(0, 4, true), 1.0));
}

TEST(Macs, minVecMacMutual)
{
    using T = double;

    Vec3<T> cA{0.5, 0.5, 0.5};
    Vec3<T> sA{0.5, 0.5, 0.5};

    Vec3<T> cB{3.5, 3.5, 3.5};
    Vec3<T> sB{0.5, 0.5, 0.5};

    EXPECT_TRUE(minVecMacMutual(cA, sA, cB, sB, Box<T>(0, 4, false), 1.0 / 0.39));
    EXPECT_FALSE(minVecMacMutual(cA, sA, cB, sB, Box<T>(0, 4, false), 1.0 / 0.38));

    EXPECT_FALSE(minVecMacMutual(cA, sA, cB, sB, Box<T>(0, 4, true), 1.0));
}

template<class KeyType, class T>
static std::vector<char> markMacAll2All(
    const Octree<KeyType>& octree, TreeNodeIndex firstLeaf, TreeNodeIndex lastLeaf, float invTheta, const Box<T>& box)
{
    gsl::span<const KeyType> leaves = octree.treeLeaves();
    std::vector<char> markings(octree.numTreeNodes(), 0);

    // loop over target cells
    for (TreeNodeIndex i = firstLeaf; i < lastLeaf; ++i)
    {
        IBox targetBox = sfcIBox(sfcKey(leaves[i]), sfcKey(leaves[i + 1]));
        auto [targetCenter, targetSize] = centerAndSize<KeyType>(targetBox, box);

        // loop over source cells
        for (TreeNodeIndex j = 0; j < octree.numTreeNodes(); ++j)
        {
            // source cells must not be in target cell range
            if (leaves[firstLeaf] <= octree.codeStart(j) && octree.codeEnd(j) <= leaves[lastLeaf])
            {
                continue;
            }
            IBox sourceBox                  = sfcIBox(sfcKey(octree.codeStart(j)), sfcKey(octree.codeEnd(j)));
            auto [sourceCenter, sourceSize] = centerAndSize<KeyType>(sourceBox, box);

            // if source cell fails MAC w.r.t to current target, it gets marked
            bool violatesMac = !minMac(targetCenter, targetSize, sourceCenter, sourceSize, box, invTheta);
            if (violatesMac) { markings[j] = 1; }
        }
    }

    return markings;
}

template<class KeyType>
static void markMac()
{
    Box<double> box(0, 1);
    std::vector<KeyType> leaves = OctreeMaker<KeyType>{}.divide().divide(0).divide(5).makeTree();

    Octree<KeyType> octree;
    octree.update(leaves.data(), nNodes(leaves));

    std::vector<char> markings(octree.numTreeNodes(), 0);

    float theta = 0.58;
    TreeNodeIndex focusIdxStart = 0;
    TreeNodeIndex focusIdxEnd   = 2;
    markMac(octree, box, leaves[focusIdxStart], leaves[focusIdxEnd], 1. / theta, markings.data());

    std::vector<char> reference = markMacAll2All<KeyType>(octree, focusIdxStart, focusIdxEnd, 1. / theta, box);

    EXPECT_EQ(std::count(begin(markings), end(markings), 0), 9);
    EXPECT_EQ(markings, reference);
}

TEST(Macs, markMac)
{
    markMac<unsigned>();
    markMac<uint64_t>();
}

template<class KeyType, class T>
static std::vector<char> markVecMacAll2All(const Octree<KeyType>& octree,
                                           const Vec4<T>* centers,
                                           TreeNodeIndex firstLeaf,
                                           TreeNodeIndex lastLeaf,
                                           const Box<T>& box)
{
    gsl::span<const KeyType> leaves = octree.treeLeaves();
    std::vector<char> markings(octree.numTreeNodes(), 0);

    // loop over target cells
    for (TreeNodeIndex i = firstLeaf; i < lastLeaf; ++i)
    {
        IBox targetBox = sfcIBox(sfcKey(leaves[i]), sfcKey(leaves[i + 1]));
        auto [targetCenter, targetSize] = centerAndSize<KeyType>(targetBox, box);

        // loop over source cells
        for (TreeNodeIndex j = 0; j < octree.numTreeNodes(); ++j)
        {
            // source cells must not be in target cell range
            if (leaves[firstLeaf] <= octree.codeStart(j) && octree.codeEnd(j) <= leaves[lastLeaf])
            {
                continue;
            }

            Vec4<T> center   = centers[j];
            bool violatesMac = vectorMacPbc(makeVec3(center), center[3], targetCenter, targetSize, box);
            if (violatesMac) { markings[j] = 1; }
        }
    }

    return markings;
}

template<class KeyType>
static void markMacVector()
{
    using T                 = double;
    LocalIndex numParticles = 1000;
    unsigned bucketSize     = 2;
    float theta             = 0.58;
    Box<T> box(0, 1);

    RandomGaussianCoordinates<T, SfcKind<KeyType>> coords(numParticles, box);
    std::vector<T> masses(numParticles, 1.0 / numParticles);

    auto [leaves, counts] = computeOctree(coords.particleKeys().data(),
                                          coords.particleKeys().data() + coords.particleKeys().size(), bucketSize);
    Octree<KeyType> octree;
    octree.update(leaves.data(), nNodes(leaves));

    std::vector<SourceCenterType<T>> centers(octree.numTreeNodes());
    computeLeafMassCenter<T, T, T, KeyType>(coords.x(), coords.y(), coords.z(), masses, coords.particleKeys(), octree,
                                            centers);
    upsweep(octree, centers.data(), CombineSourceCenter<T>{});
    setMac<T, KeyType>(octree.nodeKeys(), centers, 1.0 / theta, box);

    std::vector<char> markings(octree.numTreeNodes(), 0);

    TreeNodeIndex focusIdxStart = 4;
    TreeNodeIndex focusIdxEnd   = 22;

    markVecMac(octree, centers.data(), box, leaves[focusIdxStart], leaves[focusIdxEnd], markings.data());

    std::vector<char> reference = markVecMacAll2All<KeyType>(octree, centers.data(), focusIdxStart, focusIdxEnd, box);

    EXPECT_EQ(markings, reference);
}

TEST(Macs, markMacVector)
{
    markMacVector<unsigned>();
    markMacVector<uint64_t>();
}

} // namespace cstone
