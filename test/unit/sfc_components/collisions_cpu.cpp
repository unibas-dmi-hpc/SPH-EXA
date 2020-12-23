
#include "gtest/gtest.h"

#include "sfc/collisions_cpu.hpp"
#include "sfc/octree_util.hpp"

namespace sphexa
{
/*! \brief all-to-all implementation of findCollisions
 *
 * @tparam I   32- or 64-bit unsigned integer
 * @param[in]  tree           octree leaf nodes in cornerstone format
 * @param[out] collisionList  output list of indices of colliding nodes
 * @param[in]  collisionBox   query box to look for collisions
 *                            with leaf nodes
 *
 * Naive implementation without tree traversal for reference
 * and testing purposes
 */
template <class I>
void findCollisions2All(const std::vector<I>& tree, CollisionList& collisionList,
                        const Box<int>& collisionBox)
{
    for (std::size_t nodeIndex = 0; nodeIndex < nNodes(tree); ++nodeIndex)
    {
        int prefixBits = treeLevel(tree[nodeIndex+1] - tree[nodeIndex]) * 3;
        if (overlap(tree[nodeIndex], prefixBits, collisionBox))
            collisionList.add((int)nodeIndex);
    }
}

}

using namespace sphexa;


/*! \brief test the naive to-all collision detection function
 *
 * @tparam I  32- or 64-bit unsigned integer
 */
template<class I>
void collide2all()
{
    using sphexa::detail::codeFromIndices;

    auto tree = OctreeMaker<I>{}.divide().divide(0).divide(0,7).makeTree();

    // this search box intersects with neighbors in x direction and will intersect
    // with multiple smaller level 2 and level 3 nodes
    // it corresponds to the node with code codeFromIndices<I>({4}) with a dx=1 halo extension
    int r = 1u<<(maxTreeLevel<I>{} - 1);
    Box<int> haloBox{r-1, 2*r, 0, r, 0, r};

    CollisionList collisionList;
    findCollisions2All(tree, collisionList, haloBox);

    std::vector<I> collisions(collisionList.size());
    for (int i = 0; i < collisions.size(); ++i)
        collisions[i] = tree[collisionList[i]];

    // list of octree leaf morton codes that should collide
    // with the halo box
    std::vector<I> refCollisions{
        codeFromIndices<I>({0,4}),
        codeFromIndices<I>({0,5}),
        codeFromIndices<I>({0,6}),
        codeFromIndices<I>({0,7,4}),
        codeFromIndices<I>({0,7,5}),
        codeFromIndices<I>({0,7,6}),
        codeFromIndices<I>({0,7,7}),
        codeFromIndices<I>({4})
    };

    EXPECT_EQ(collisions, refCollisions);
}

TEST(Collisions, collide2all)
{
    collide2all<unsigned>();
    collide2all<uint64_t>();
}
