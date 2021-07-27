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
 * @brief Binary radix tree traversal tests with explicit references
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <vector>

#include "gtest/gtest.h"

#include "cstone/halos/btreetraversal.hpp"
#include "cstone/tree/octree.hpp"
#include "cstone/tree/octree_util.hpp"

using namespace cstone;

//! @brief test add(), operator[] and begin()/end() of the CollisionList class
TEST(BinaryTreeTraversal, collisionList)
{
    CollisionList collisions;
    collisions.add(3);
    collisions.add(7);
    collisions.add(10);
    collisions.add(0);

    EXPECT_EQ(collisions.size(), 4);
    EXPECT_EQ(collisions[0], 3);
    EXPECT_EQ(collisions[1], 7);
    EXPECT_EQ(collisions[2], 10);
    EXPECT_EQ(collisions[3], 0);

    std::vector<int> refValues{3, 7, 10, 0};
    std::vector<int> probe{collisions.begin(), collisions.end()};

    EXPECT_EQ(refValues, probe);
}

/*! @brief test findCollisions with pbc halo boxes
 *
 * The example constructs a 4x4x4 regular octree with 64 nodes. A haloBox
 * with coordinates [-1,1]^3 will collide with all 8 nodes in the corners of the tree.
 * The same happens for a haloBox at the opposite diagonal end with
 * coordinates [2^(10 or 21)-1, 2^(10 or 21)+1]^3.
 */
template<class I>
void pbcCollision()
{
    std::vector<I> tree = makeUniformNLevelTree<I>(64, 1);

    std::vector<BinaryNode<I>> internalTree(nNodes(tree));
    createBinaryTree(tree.data(), nNodes(tree), internalTree.data());

    // halo box around coordinate origin
    {
        IBox haloBox{-1, 1, -1, 1, -1, 1};

        CollisionList collisions;
        findCollisions(internalTree.data(), tree.data(), collisions, haloBox, {0, 0});

        std::vector<int> collided(collisions.begin(), collisions.end());
        std::sort(begin(collided), end(collided));

        constexpr int level2 = 2;
        // nLevel2Codes is the number of Morton codes that a level2 node can cover,
        // corresponding to the number of contained octree leaf nodes at the maximum division level
        I nLevel2Codes = 1ul << (3 * (maxTreeLevel<I>{} - level2)); // 8^(maxLevel - 2)
        // reference from level-2 ix,iy,iz coordinates in [0:4], illustrates that all 8 corners collide
        std::vector<int> refCollided{
            int(iMorton<I>(0, 0, 0, level2) / nLevel2Codes), int(iMorton<I>(0, 0, 3, level2) / nLevel2Codes),
            int(iMorton<I>(0, 3, 0, level2) / nLevel2Codes), int(iMorton<I>(0, 3, 3, level2) / nLevel2Codes),
            int(iMorton<I>(3, 0, 0, level2) / nLevel2Codes), int(iMorton<I>(3, 0, 3, level2) / nLevel2Codes),
            int(iMorton<I>(3, 3, 0, level2) / nLevel2Codes), int(iMorton<I>(3, 3, 3, level2) / nLevel2Codes),
        };
        // node indices of the 8 corners in a level-2 tree with 4x4x4=64 nodes
        // explicitly specified
        std::vector<int> refCollidedExplicit{0, 9, 18, 27, 36, 45, 54, 63};

        EXPECT_EQ(collided, refCollided);
        EXPECT_EQ(refCollided, refCollidedExplicit);
    }

    // halo box around far corner
    {
        int maxCoord = 1u << maxTreeLevel<I>{};
        IBox haloBox{maxCoord - 1, maxCoord + 1, maxCoord - 1, maxCoord + 1, maxCoord - 1, maxCoord + 1};

        CollisionList collisions;
        findCollisions(internalTree.data(), tree.data(), collisions, haloBox, {0, 0});

        std::vector<int> collided(collisions.begin(), collisions.end());
        std::sort(begin(collided), end(collided));
        // 8 corners
        std::vector<int> refCollided{0, 9, 18, 27, 36, 45, 54, 63};

        EXPECT_EQ(collided, refCollided);
    }
}

TEST(BinaryTreeTraversal, pbcCollision)
{
    pbcCollision<unsigned>();
    pbcCollision<uint64_t>();
}

/*! @brief test findCollisions with pbc halo boxes and part of the tree excluded from collision detection
 *
 * The example constructs a 4x4x4 regular octree with 64 nodes. A haloBox
 * with coordinates [-1,1]^3 will collide with all 8 nodes in the corners of the tree, except if excluded.
 * The same happens for a haloBox at the opposite diagonal end with
 * coordinates [2^(10 or 21)-1, 2^(10 or 21)+1]^3.
 */
template<class I>
void pbcCollisionWithExclusions()
{
    constexpr int level2 = 2;
    std::vector<I> tree = makeUniformNLevelTree<I>(64, 1);

    std::vector<BinaryNode<I>> internalTree(nNodes(tree));
    createBinaryTree(tree.data(), nNodes(tree), internalTree.data());

    IBox haloBox{-1, 1, -1, 1, -1, 1};
    {
        CollisionList collisions;
        findCollisions(internalTree.data(), tree.data(), collisions, haloBox, {0, nodeRange<I>(level2)});

        std::vector<int> collided(collisions.begin(), collisions.end());
        std::sort(begin(collided), end(collided));

        // node 0 gets excluded
        std::vector<int> refCollided{0, 9, 18, 27, 36, 45, 54, 63};

        EXPECT_EQ(collided, refCollided);
    }
    {
        CollisionList collisions;
        findCollisions(internalTree.data(), tree.data(), collisions, haloBox,
                       {iMorton<I>(0, 0, 3, level2), iMorton<I>(0, 0, 3, level2) + nodeRange<I>(level2)});

        std::vector<int> collided(collisions.begin(), collisions.end());
        std::sort(begin(collided), end(collided));

        // node 9 gets excluded
        std::vector<int> refCollided{0, 18, 27, 36, 45, 54, 63};

        EXPECT_EQ(collided, refCollided);
    }
    {
        CollisionList collisions;
        findCollisions(internalTree.data(), tree.data(), collisions, haloBox,
                       {iMorton<I>(3, 3, 3, level2), iMorton<I>(3, 3, 3, level2) + nodeRange<I>(level2)});

        std::vector<int> collided(collisions.begin(), collisions.end());
        std::sort(begin(collided), end(collided));

        // node 63 gets excluded
        std::vector<int> refCollided{0, 9, 18, 27, 36, 45, 54};

        EXPECT_EQ(collided, refCollided);
    }
}

TEST(BinaryTreeTraversal, pbcCollisionWithExclusions)
{
    pbcCollision<unsigned>();
    pbcCollision<uint64_t>();
}

/*! @brief test collision detection with anisotropic halo ranges
 *
 * If the bounding box of the floating point boundary box is not cubic,
 * an isotropic search range with one halo radius per node will correspond
 * to an anisotropic range in the Morton code SFC which always gets mapped
 * to an unit cube.
 */
template<class I>
void anisotropicHaloBox()
{
    // a tree with 4 subdivisions along each dimension, 64 nodes
    // node range in each dimension is 2^(10 or 21 - 2)
    std::vector<I> tree = makeUniformNLevelTree<I>(64, 1);

    std::vector<BinaryNode<I>> internalTree(nNodes(tree));
    createBinaryTree(tree.data(), nNodes(tree), internalTree.data());

    int r = 1u << (maxTreeLevel<I>{} - 2);

    int queryIdx = 7;

    // this will hit two nodes in +x direction, not just one neighbor node
    IBox haloBox = makeHaloBox(tree[queryIdx], tree[queryIdx + 1], 2 * r, 0, 0);

    CollisionList collisions;
    findCollisions(internalTree.data(), tree.data(), collisions, haloBox, {0, 0});

    std::vector<int> collisionsSorted(collisions.begin(), collisions.end());
    std::sort(begin(collisionsSorted), end(collisionsSorted));

    std::vector<int> collisionsReference{3, 7, 35, 39};
    EXPECT_EQ(collisionsSorted, collisionsReference);
}

TEST(BinaryTreeTraversal, anisotropicHalo)
{
    anisotropicHaloBox<unsigned>();
    anisotropicHaloBox<uint64_t>();
}

//! @brief this tree results from 2 particles at (0,0,0) and at (1,1,1) with a bucket size of 1
std::vector<unsigned> makeEdgeTree()
{
    std::vector<unsigned> tree{
        0,          1,          2,          3,          4,          5,          6,          7,          8,
        16,         24,         32,         40,         48,         56,         64,         128,        192,
        256,        320,        384,        448,        512,        1024,       1536,       2048,       2560,
        3072,       3584,       4096,       8192,       12288,      16384,      20480,      24576,      28672,
        32768,      65536,      98304,      131072,     163840,     196608,     229376,     262144,     524288,
        786432,     1048576,    1310720,    1572864,    1835008,    2097152,    4194304,    6291456,    8388608,
        10485760,   12582912,   14680064,   16777216,   33554432,   50331648,   67108864,   83886080,   100663296,
        117440512,  134217728,  268435456,  402653184,  536870912,  671088640,  805306368,  939524096,  956301312,
        973078528,  989855744,  1006632960, 1023410176, 1040187392, 1056964608, 1059061760, 1061158912, 1063256064,
        1065353216, 1067450368, 1069547520, 1071644672, 1071906816, 1072168960, 1072431104, 1072693248, 1072955392,
        1073217536, 1073479680, 1073512448, 1073545216, 1073577984, 1073610752, 1073643520, 1073676288, 1073709056,
        1073713152, 1073717248, 1073721344, 1073725440, 1073729536, 1073733632, 1073737728, 1073738240, 1073738752,
        1073739264, 1073739776, 1073740288, 1073740800, 1073741312, 1073741376, 1073741440, 1073741504, 1073741568,
        1073741632, 1073741696, 1073741760, 1073741768, 1073741776, 1073741784, 1073741792, 1073741800, 1073741808,
        1073741816, 1073741817, 1073741818, 1073741819, 1073741820, 1073741821, 1073741822, 1073741823, 1073741824};

    return tree;
}

/*! @brief a simple collision test with the edge tree from above
 *
 * Since the halo radius for the first and last node is bigger than the box,
 * these two nodes collide with all nodes in the tree, while all other nodes have
 * radius 0 and only collide with themselves.
 */
TEST(Collisions, adjacentEdgeRegression)
{
    std::vector<unsigned> tree = makeEdgeTree();
    std::vector<BinaryNode<unsigned>> internalTree(nNodes(tree));
    createBinaryTree(tree.data(), nNodes(tree), internalTree.data());

    Box<double> box(0.5, 0.6);

    std::vector<double> haloRadii(nNodes(tree), 0);
    haloRadii[0] = 0.2;
    *haloRadii.rbegin() = 0.2;

    std::vector<int> allNodes(nNodes(tree));
    std::iota(begin(allNodes), end(allNodes), 0);

    for (std::size_t i = 0; i < nNodes(tree); ++i)
    {
        IBox haloBox = makeHaloBox(tree[i], tree[i + 1], haloRadii[i], box);
        CollisionList collisions;
        findCollisions(internalTree.data(), tree.data(), collisions, haloBox, {0, 0});

        std::vector<int> cnodes{collisions.begin(), collisions.end()};
        std::sort(begin(cnodes), end(cnodes));

        if (i == 0 || i == nNodes(tree) - 1) { EXPECT_EQ(cnodes, allNodes); }
        else
        {
            EXPECT_EQ(cnodes, std::vector<int>(1, i));
        }
    }
}

/*! @brief collisions test with a very small radius
 *
 * This tests that a very small, but non-zero halo radius
 * does not get rounded down to zero.
 */
TEST(Collisions, adjacentEdgeSmallRadius)
{
    std::vector<unsigned> tree = makeEdgeTree();
    std::vector<BinaryNode<unsigned>> internalTree(nNodes(tree));
    createBinaryTree(tree.data(), nNodes(tree), internalTree.data());

    Box<double> box(0, 1);

    // nNodes is 134
    int secondLastNode = 132;
    double radius = 0.0001;
    IBox haloBox = makeHaloBox(tree[secondLastNode], tree[secondLastNode + 1], radius, box);

    CollisionList collisions;
    findCollisions(internalTree.data(), tree.data(), collisions, haloBox, {0, 0});

    std::vector<int> cnodes{collisions.begin(), collisions.end()};
    std::sort(begin(cnodes), end(cnodes));

    std::vector<int> refNodes{125, 126, 127, 128, 129, 130, 131, 132, 133};
    EXPECT_EQ(cnodes, refNodes);
}

TEST(Collisions, adjacentEdgeLastNode)
{
    std::vector<unsigned> tree = makeEdgeTree();
    std::vector<BinaryNode<unsigned>> internalTree(nNodes(tree));
    createBinaryTree(tree.data(), nNodes(tree), internalTree.data());

    Box<double> box(0, 1);

    // nNodes is 134
    int lastNode = 133;
    double radius = 0.0;
    IBox haloBox = makeHaloBox(tree[lastNode], tree[lastNode + 1], radius, box);

    CollisionList collisions;
    findCollisions(internalTree.data(), tree.data(), collisions, haloBox, {0, 0});

    std::vector<int> cnodes{collisions.begin(), collisions.end()};
    std::sort(begin(cnodes), end(cnodes));

    std::vector<int> refNodes{133};
    EXPECT_EQ(cnodes, refNodes);
}

TEST(Collisions, regularLastNode)
{
    // a tree with 4 subdivisions along each dimension, 64 nodes
    std::vector<unsigned> tree = makeUniformNLevelTree<unsigned>(64, 1);
    std::vector<BinaryNode<unsigned>> internalTree(nNodes(tree));
    createBinaryTree(tree.data(), nNodes(tree), internalTree.data());

    IBox haloBox{1022, 1023, 1022, 1023, 1022, 1023};
    CollisionList collisions;
    findCollisions(internalTree.data(), tree.data(), collisions, haloBox, {0, 0});
    EXPECT_EQ(collisions.size(), 1);
    EXPECT_EQ(collisions[0], 63);
}
