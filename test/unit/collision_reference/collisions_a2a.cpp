
#include "gtest/gtest.h"

#include "collisions_a2a.hpp"


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

/*! \brief test the naive all-to-all collision detection function
 *
 * @tparam I  32- or 64-bit unsigned integer
 */
template<class I, class T>
void collideAll2all()
{
    using sphexa::detail::codeFromIndices;

    auto tree = OctreeMaker<I>{}.divide().divide(0).divide(0,7).makeTree();

    Box<T> box(0, 1);
    std::vector<T> haloRadii(nNodes(tree), 0.1);

    std::vector<CollisionList> allCollisions = findCollisionsAll2all(tree, haloRadii, box);

    // extract list of collisions for node with index 18, corresponding to {4}
    std::vector<I> n18coll(allCollisions[18].size());
    for (int i = 0; i < n18coll.size(); ++i)
        n18coll[i] = tree[allCollisions[18][i]];

    std::sort(begin(n18coll), end(n18coll));

    // reference list of collisions for node with index 18, corresponding to {4}
    std::vector<I> refCollisions{
        codeFromIndices<I>({0,4}),
        codeFromIndices<I>({0,5}),
        codeFromIndices<I>({0,6}),
        codeFromIndices<I>({0,7,4}),
        codeFromIndices<I>({0,7,5}),
        codeFromIndices<I>({0,7,6}),
        codeFromIndices<I>({0,7,7}),
        codeFromIndices<I>({1}),
        codeFromIndices<I>({2}),
        codeFromIndices<I>({3}),
        codeFromIndices<I>({4}),
        codeFromIndices<I>({5}),
        codeFromIndices<I>({6}),
        codeFromIndices<I>({7})
    };

    EXPECT_EQ(n18coll, refCollisions);
}

TEST(Collisions, collideAll2all)
{
    collideAll2all<unsigned, float>();
    collideAll2all<uint64_t, float>();
    collideAll2all<unsigned, double>();
    collideAll2all<uint64_t, double>();
}
