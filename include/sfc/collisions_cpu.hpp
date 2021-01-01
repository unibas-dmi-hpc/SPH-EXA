#pragma once

#include "sfc/btreetraversal.hpp"

namespace sphexa
{

/*! \brief For each leaf node enlarged by its halo radius, find all colliding leaf nodes
 *
 * @tparam I            32- or 64-bit unsigned integer
 * @param internalTree  internal binary tree
 * @param tree          sorted Morton codes representing the leaves of the (global) octree
 * @param haloRadii     halo search radii per leaf node, length = nNodes(tree)
 * @return              list of colliding node indices for each leaf node
 *
 * This is a CPU version that can be OpenMP parallelized.
 * In the GPU version, the for-loop body is designed such that one GPU-thread
 * can be launched for each for-loop element.
 */
template<class I, class T>
std::vector<CollisionList> findAllCollisions(const std::vector<BinaryNode<I>>& internalTree, const std::vector<I>& tree,
                                             const std::vector<T>& haloRadii, const Box<T>& globalBox)
{
    assert(internalTree.size() == tree.size() - 1 && "internal tree does not match leaves");
    assert(internalTree.size() == haloRadii.size() && "need one halo radius per leaf node");

    std::vector<CollisionList> collisions(tree.size() - 1);

    // (omp) parallel
    for (int leafIdx = 0; leafIdx < internalTree.size(); ++leafIdx)
    {
        T radius = haloRadii[leafIdx];

        Box<int> haloBox = makeHaloBox(tree[leafIdx], tree[leafIdx+1], radius, globalBox);
        findCollisions(internalTree.data(), tree.data(), collisions[leafIdx], haloBox);
    }

    return collisions;
}

} // namespace sphexa