#include <vector>

#include "gtest/gtest.h"

#include "sfc/btreetraversal.hpp"
#include "sfc/octree.hpp"

using namespace sphexa;

//! \brief test add(), operator[] and begin()/end() of the CollisionList class
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

    std::vector<int> refValues{3,7,10,0};
    std::vector<int> probe{collisions.begin(), collisions.end()};

    EXPECT_EQ(refValues, probe);
}

/*! \brief test findCollisions with halo boxes that exceed the Morton range
 *
 * The Morton integer coordinate ranges in x,y,z are either [0,2^10]
 * or [0,2^21]. But findCollisions should be able to correctly report
 * no collisions if the halo box lies outside the supported range.
 */
template<class I>
void boxOutOfRange()
{
    std::vector<I>             tree         = makeUniformNLevelTree<I>(8, 1);
    std::vector<BinaryNode<I>> internalTree = createInternalTree(tree);

    int r = 1u<<maxTreeLevel<I>{};

    {
        // box exceeds maximum integer coordinates by +-1 in all dimensions
        Box<int> haloBox{-1, r, -1, r, -1, r};

        CollisionList collisions;
        findCollisions(internalTree.data(), tree.data(), collisions, haloBox);

        // all nodes should collide
        EXPECT_EQ(collisions.size(), nNodes(tree));
    }
    {
        // box exceeds maximum integer coordinates by +-1 in all dimensions
        Box<int> haloBox{r, r+1, 0, r, 0, r};

        CollisionList collisions;
        findCollisions(internalTree.data(), tree.data(), collisions, haloBox);

        // no nodes collide
        EXPECT_EQ(collisions.size(), 0);
    }
}

TEST(BinaryTreeTraversal, boxOutOfRange)
{
    boxOutOfRange<unsigned>();
    boxOutOfRange<uint64_t>();
}


/*! \brief test collision detection with anisotropic halo ranges
 *
 * If the bounding box of the floating point boundary box is not cubic,
 * an isotropic search range with one halo radius per node will correspond
 * to an anisotropic range in the Morton code SFC which always gets mapped
 * to an unit cube.
 */
template <class I>
void anisotropicHaloBox()
{
    // a tree with 4 subdivisions along each dimension, 64 nodes
    // node range in each dimension is 2^(10 or 21 - 2)
    std::vector<I>             tree         = makeUniformNLevelTree<I>(64, 1);
    std::vector<BinaryNode<I>> internalTree = createInternalTree(tree);

    int r = 1u<<(maxTreeLevel<I>{}-2);

    int queryIdx = 7;

    // this will hit two nodes in +x direction, not just one neighbor node
    Box<int> haloBox = makeHaloBox(tree[queryIdx], tree[queryIdx+1], 2*r, 0, 0);

    CollisionList collisions;
    findCollisions(internalTree.data(), tree.data(), collisions, haloBox);

    std::vector<int> collisionsSorted(collisions.begin(), collisions.end());
    std::sort(begin(collisionsSorted), end(collisionsSorted));

    std::vector<int> collisionsReference{3,7,35,39};
    EXPECT_EQ(collisionsSorted, collisionsReference);
}


TEST(BinaryTreeTraversal, anisotropicHalo)
{
    anisotropicHaloBox<unsigned>();
    anisotropicHaloBox<uint64_t>();
}


