#include <vector>

#include "gtest/gtest.h"

#include "sfc/btreetraversal.hpp"
#include "sfc/collisions_cpu.hpp"
#include "sfc/octree.hpp"
#include "sfc/octree_util.hpp"

using namespace sphexa;

/*! \brief Traversal test for all leaves in a regular octree
 *
 * This test performs the following:
 *
 * 1. Create the leaves of a regular octree with 64 leaves and the
 *    corresponding internal binary part.
 *
 * 2. For each leaf enlarged by the halo range, find collisions
 *    between all the other leaves.
 *
 * 3. a) For each leaf, compute x,y,z coordinate ranges of the leaf + halo radius
 *    b) Test all the other leaves for overlap with the ranges of part a)
 *       If a collision between the node pair was reported in 2., there has to be overlap,
 *       if no collision was reported, there must not be any overlap.
 */
template <class I>
void regular4x4x4traversalTest()
{
    /// 1.
    // a tree with 4 subdivisions along each dimension, 64 nodes
    // node range in each dimension is 256
    std::vector<I> tree = makeUniformNLevelTree<I>(64, 1);

    auto internalTree = createInternalTree(tree);

    // halo ranges
    int dx = 1;
    int dy = 1;
    int dz = 1;

    // if the box has size [0, 2^10-1]^3 (32-bit) or [0, 2^21]^3 (64-bit),
    // radius (1 + epsilon) in double will translate to radius 1 normalized to integer.
    Box<double> box(0, (1u<<maxTreeLevel<I>{})-1);
    std::vector<double> haloRadii(nNodes(tree), 1.1);

    EXPECT_EQ(dx, sphexa::detail::toNBitInt<I>(normalize(haloRadii[0], box.xmin(), box.xmax())));
    EXPECT_EQ(dy, sphexa::detail::toNBitInt<I>(normalize(haloRadii[0], box.ymin(), box.ymax())));
    EXPECT_EQ(dz, sphexa::detail::toNBitInt<I>(normalize(haloRadii[0], box.zmin(), box.zmax())));

    /// 2.
    // find collisions of all leaf nodes enlarged by the halo ranges with all the other leaves
    // with (dx,dy,dz) = (1,1,1), this finds all immediate neighbors
    std::vector<CollisionList> collisions = findAllCollisions(internalTree, tree, haloRadii, box);

    /// 3. a)
    for (int leafIdx = 0; leafIdx < nNodes(tree); ++leafIdx)
    {
        Box<int> haloBox = makeHaloBox(tree[leafIdx], tree[leafIdx+1], dx, dy, dz);

        // number of nearest neighbors in a regular 3D grid is between 8 and 27
        EXPECT_GE(collisions[leafIdx].size(), 8);
        EXPECT_LE(collisions[leafIdx].size(), 27);

        /// 3. b)
        for (int cIdx = 0; cIdx < nNodes(tree); ++cIdx)
        {
            int collisionNodeIndex   = collisions[leafIdx][cIdx];
            I collisionLeafCode      = tree[collisionNodeIndex];
            I collisionLeafCodeUpper = tree[collisionNodeIndex+1];
            int nBits = treeLevel(collisionLeafCodeUpper - collisionLeafCode) * 3;

            // has a collision been reported between leafNodes cIdx and leafIdx?
            bool hasCollision =
                std::find(collisions[leafIdx].begin(), collisions[leafIdx].end(), collisionNodeIndex) != collisions[leafIdx].end();

            if (hasCollision)
            {
                // if yes, then the cIdx nodes has to overlap with leafIdx enlarged by the halos
                EXPECT_TRUE(overlap(collisionLeafCode, nBits, haloBox));
            }
            else
            {
                // if not, then there must not be any overlap
                EXPECT_FALSE(overlap(collisionLeafCode, nBits, haloBox));
            }
        }
    }
}

TEST(BinaryTreeTraversal, regularTree4x4x4FullTraversal)
{
    regular4x4x4traversalTest<unsigned>();
    regular4x4x4traversalTest<uint64_t>();
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


template<class I>
void irregularTreeTraversal()
{
    using sphexa::detail::codeFromIndices;

    auto tree = OctreeMaker<I>{}.divide().divide(0).divide(0,7).makeTree();

    std::vector<BinaryNode<I>> internalTree = createInternalTree(tree);

    // quick test that we can locate nodes in the {l1,l2,l3,...} format with std::find
    {
        int idx1 = std::find(begin(tree), end(tree), codeFromIndices<I>({0,7,0}))
                   - begin(tree);
        EXPECT_EQ(7, idx1);

        int idx2 = std::find(begin(tree), end(tree), codeFromIndices<I>({0,7,7}))
                   - begin(tree);
        EXPECT_EQ(14, idx2);
    }

    // launch collision detection with a big level 1 node next to the small level 3 ones
    {
        I queryNode = codeFromIndices<I>({4});
        int queryIdx = std::find(begin(tree), end(tree), queryNode) - begin(tree);
        EXPECT_EQ(18, queryIdx);

        // this halo box intersects with neighbors in x direction and will intersect
        // with multiple smaller level 2 and level 3 nodes
        Box<int> haloBox = makeHaloBox(tree[queryIdx], tree[queryIdx + 1], 1, 0, 0);

        CollisionList collisions;
        findCollisions(internalTree.data(), tree.data(), collisions, haloBox);

        // list of nodes that should collide with the halo box
        std::vector<I> collidingNodes{
            codeFromIndices<I>({0,4}),
            codeFromIndices<I>({0,5}),
            codeFromIndices<I>({0,6}),
            codeFromIndices<I>({0,7,4}),
            codeFromIndices<I>({0,7,5}),
            codeFromIndices<I>({0,7,6}),
            codeFromIndices<I>({0,7,7}),
            codeFromIndices<I>({4}),
        };

        //for (I code : collidingNodes)
        //{
        //    int index = std::find(begin(tree), end(tree), code) - begin(tree);
        //    bool collided = std::find(collisions.begin(), collisions.end(), index)
        //                        != collisions.end();

        //    bool olp = false;
        //    {
        //        int prefixBits = treeLevel(tree[index+1] - tree[index]) * 3;
        //        olp = overlap(tree[index], prefixBits, haloBox);
        //    }

        //    std::cout << collided << " " << olp << std::endl;
        //}

        EXPECT_EQ(collisions.size(), collidingNodes.size());

        // go through all leaf nodes and check that collisions have only
        // been reported for the nodes listed in collidingNodes
        for(int nodeIdx = 0; nodeIdx < nNodes(tree); ++nodeIdx)
        {
            bool shouldCollide = std::find(begin(collidingNodes), end(collidingNodes), tree[nodeIdx])
                                 != end(collidingNodes);
            bool didCollide = std::find(collisions.begin(), collisions.end(), nodeIdx)
                              != collisions.end();

            EXPECT_EQ(shouldCollide, didCollide);
        }
    }
}

TEST(BinaryTreeTraversal, irregularTreeTraversal)
{
    irregularTreeTraversal<unsigned>();
    irregularTreeTraversal<uint64_t>();
}
