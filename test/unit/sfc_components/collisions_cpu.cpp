
#include "gtest/gtest.h"

#include "sfc/collisions_cpu.hpp"
#include "sfc/octree_util.hpp"

namespace sphexa
{
/*! \brief to-all implementation of findCollisions
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

//! \brief all-to-all implementation of findAllCollisions
template<class I, class T>
std::vector<CollisionList> findCollisionsAll2all(const std::vector<I>& tree, const std::vector<T>& haloRadii,
                                                 const Box<T>& globalBox)
{
    std::vector<CollisionList> collisions(tree.size() - 1);

    for (int leafIdx = 0; leafIdx < nNodes(tree); ++leafIdx)
    {
        T radius = haloRadii[leafIdx];

        int dx = detail::toNBitInt<I>(normalize(radius, globalBox.xmin(), globalBox.xmax()));
        int dy = detail::toNBitInt<I>(normalize(radius, globalBox.ymin(), globalBox.ymax()));
        int dz = detail::toNBitInt<I>(normalize(radius, globalBox.zmin(), globalBox.zmax()));

        Box<int> haloBox = makeHaloBox(tree[leafIdx], tree[leafIdx + 1], dx, dy, dz);
        findCollisions2All(tree, collisions[leafIdx], haloBox);
    }

    return collisions;
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


//! \brief compare tree-traversal collision detection with the reference implementation
template<class I, class T>
void generalCollisionTest(const std::vector<I>& tree, const std::vector<T>& haloRadii,
                          const Box<T>& box)
{
    std::vector<BinaryNode<I>> internalTree = createInternalTree(tree);

    // tree traversal collision detection
    std::vector<CollisionList> collisions    = findAllCollisions(internalTree, tree, haloRadii, box);
    // reference implementation
    std::vector<CollisionList> refCollisions = findCollisionsAll2all(tree, haloRadii, box);

    for (int nodeIndex = 0; nodeIndex < nNodes(tree); ++nodeIndex)
    {
        std::vector<int> c{collisions[nodeIndex].begin(), collisions[nodeIndex].end()};
        std::vector<int> ref{refCollisions[nodeIndex].begin(), refCollisions[nodeIndex].end()};

        std::sort(begin(c), end(c));
        std::sort(begin(ref), end(ref));

        EXPECT_EQ(c, ref);
    }
}

template<class I, class T>
void irregularTreeTraversal()
{
    auto tree = OctreeMaker<I>{}.divide().divide(0).divide(0,7).makeTree();

    Box<T> box(0, 1);
    std::vector<T> haloRadii(nNodes(tree), 0.1);
}

TEST(Collisions, irregularTreeTraversal)
{
    irregularTreeTraversal<unsigned, float>();
    irregularTreeTraversal<uint64_t, float>();
    irregularTreeTraversal<unsigned, double>();
    irregularTreeTraversal<uint64_t, double>();
}