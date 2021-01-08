#pragma once

#include "sfc/btreetraversal.hpp"
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

        int dx = detail::toNBitInt<I>(abs(normalize(radius, globalBox.xmin(), globalBox.xmax())));
        int dy = detail::toNBitInt<I>(abs(normalize(radius, globalBox.ymin(), globalBox.ymax())));
        int dz = detail::toNBitInt<I>(abs(normalize(radius, globalBox.zmin(), globalBox.zmax())));

        Box<int> haloBox = makeHaloBox(tree[leafIdx], tree[leafIdx + 1], dx, dy, dz);
        findCollisions2All(tree, collisions[leafIdx], haloBox);
    }

    return collisions;
}

} // namespace sphexa