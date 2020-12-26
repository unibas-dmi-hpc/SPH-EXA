#include <vector>

#include "gtest/gtest.h"

#include "sfc/btreetraversal.hpp"
#include "sfc/octree.hpp"

using namespace sphexa;

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


