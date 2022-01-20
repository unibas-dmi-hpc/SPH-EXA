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
 * @brief Binary radix tree traversal tests with naive all-to-all collisions as reference
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/traversal/collisions.hpp"
#include "cstone/tree/octree_util.hpp"

#include "unit/traversal/collisions_a2a.hpp"

using namespace cstone;

/*! @brief compare tree-traversal collision detection with the naive all-to-all algorithm
 *
 * @tparam KeyType     32- or 64-bit unsigned integer
 * @tparam T           float or double
 * @param  tree        cornerstone octree leaves
 * @param  haloRadii   floating point collision radius per octree leaf
 * @param  box         bounding box used to construct the octree
 *
 * This test goes through all leaf nodes of the input octree and computes
 * a list of all other leaves that overlap with the first one.
 * The computation is done with both the tree-traversal algorithm and the
 * naive all-to-all algorithm and the results are compared.
 */
template<class KeyType, class T>
static void generalCollisionTest(const std::vector<KeyType>& tree, const std::vector<T>& haloRadii, const Box<T>& box)
{
    TdOctree<KeyType> octree;
    octree.update(tree.data(), nNodes(tree));

    // tree traversal collision detection
    std::vector<std::vector<TreeNodeIndex>> collisions(nNodes(tree));
    for (std::size_t leafIdx = 0; leafIdx < nNodes(tree); ++leafIdx)
    {
        T radius = haloRadii[leafIdx];
        IBox haloBox = makeHaloBox(tree[leafIdx], tree[leafIdx + 1], radius, box);

        auto storeCollisions = [&collisionList = collisions[leafIdx]](TreeNodeIndex i) { collisionList.push_back(i); };

        findCollisions(octree, storeCollisions, haloBox, KeyType(0), KeyType(0));
    }

    // naive all-to-all algorithm
    std::vector<std::vector<TreeNodeIndex>> refCollisions = findCollisionsAll2all<KeyType>(tree, haloRadii, box);

    for (std::size_t nodeIndex = 0; nodeIndex < nNodes(tree); ++nodeIndex)
    {
        std::sort(begin(collisions[nodeIndex]), end(collisions[nodeIndex]));
        std::sort(begin(refCollisions[nodeIndex]), end(refCollisions[nodeIndex]));

        EXPECT_EQ(collisions[nodeIndex], refCollisions[nodeIndex]);
    }
}

//! @brief an irregular tree with level-3 nodes next to level-1 ones
template<class I, class T, bool Pbc>
void irregularTreeTraversal()
{
    auto tree = OctreeMaker<I>{}.divide().divide(0).divide(0, 7).makeTree();

    Box<T> box(0, 1, 0, 1, 0, 1, Pbc, Pbc, Pbc);
    std::vector<T> haloRadii(nNodes(tree), 0.1);
    generalCollisionTest(tree, haloRadii, box);
}

TEST(Collisions, irregularTreeTraversal)
{
    irregularTreeTraversal<unsigned, float, false>();
    irregularTreeTraversal<uint64_t, float, false>();
    irregularTreeTraversal<unsigned, double, false>();
    irregularTreeTraversal<uint64_t, double, false>();
}

TEST(Collisions, irregularTreeTraversalPbc)
{
    irregularTreeTraversal<unsigned, float, true>();
    irregularTreeTraversal<uint64_t, float, true>();
    irregularTreeTraversal<unsigned, double, true>();
    irregularTreeTraversal<uint64_t, double, true>();
}

//! @brief a regular tree with level-3 nodes, 8x8x8 grid
template<class I, class T, bool Pbc>
void regularTreeTraversal()
{
    auto tree = makeUniformNLevelTree<I>(512, 1);

    Box<T> box(0, 1, 0, 1, 0, 1, Pbc, Pbc, Pbc);
    // node edge length is 0.125
    std::vector<T> haloRadii(nNodes(tree), 0.124);
    generalCollisionTest(tree, haloRadii, box);
}

TEST(Collisions, regularTreeTraversal)
{
    regularTreeTraversal<unsigned, float, false>();
    regularTreeTraversal<uint64_t, float, false>();
    regularTreeTraversal<unsigned, double, false>();
    regularTreeTraversal<uint64_t, double, false>();
}

TEST(Collisions, regularTreeTraversalPbc)
{
    regularTreeTraversal<unsigned, float, true>();
    regularTreeTraversal<uint64_t, float, true>();
    regularTreeTraversal<unsigned, double, true>();
    regularTreeTraversal<uint64_t, double, true>();
}

/*! @brief test tree traversal with anisotropic boxes
 *
 * anisotropic boxes with a single halo radius per node
 * results in different x,y,z halo search lengths once
 * the coordinates are normalized to the cubic unit box.
 */
class AnisotropicBoxTraversal : public testing::TestWithParam<std::array<int, 6>>
{
public:
    template<class I, class T>
    void check()
    {
        // 8x8x8 grid
        auto tree = makeUniformNLevelTree<I>(512, 1);

        Box<T> box(std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()), std::get<3>(GetParam()),
                   std::get<4>(GetParam()), std::get<5>(GetParam()));

        // node edge length is 0.125 in the compressed dimension
        // and 0.250 in the other two dimensions
        std::vector<T> haloRadii(nNodes(tree), 0.175);
        generalCollisionTest(tree, haloRadii, box);
    }
};

TEST_P(AnisotropicBoxTraversal, compressedAxis32f) { check<unsigned, float>(); }

TEST_P(AnisotropicBoxTraversal, compressedAxis64f) { check<uint64_t, float>(); }

TEST_P(AnisotropicBoxTraversal, compressedAxis32d) { check<unsigned, double>(); }

TEST_P(AnisotropicBoxTraversal, compressedAxis64d) { check<uint64_t, double>(); }

std::vector<std::array<int, 6>> boxLimits{{0, 1, 0, 2, 0, 2}, {0, 2, 0, 1, 0, 2}, {0, 2, 0, 2, 0, 1}};

INSTANTIATE_TEST_SUITE_P(AnisotropicBoxTraversal, AnisotropicBoxTraversal, testing::ValuesIn(boxLimits));
