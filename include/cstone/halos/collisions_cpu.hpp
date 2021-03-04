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

/*! \file
 * \brief CPU driver for 3D collision detection
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "btreetraversal.hpp"
#include "traversal.hpp"

#include "cstone/tree/octree_internal.hpp"

namespace cstone
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

    #pragma omp parallel for
    for (std::size_t leafIdx = 0; leafIdx < internalTree.size(); ++leafIdx)
    {
        T radius = haloRadii[leafIdx];

        IBox haloBox = makeHaloBox(tree[leafIdx], tree[leafIdx+1], radius, globalBox);
        findCollisions(internalTree.data(), tree.data(), collisions[leafIdx], haloBox);
    }

    return collisions;
}

template<class I, class T>
std::vector<CollisionList> findAllCollisions(const Octree<I>& octree, const std::vector<T>& haloRadii, const Box<T>& globalBox)
{
    assert(octree.nLeaves() == haloRadii.size() && "need one halo radius per leaf node");

    std::vector<CollisionList> collisions(octree.nLeaves());

    #pragma omp parallel for
    for (std::size_t leafIdx = 0; leafIdx < octree.nLeaves(); ++leafIdx)
    {
        T radius = haloRadii[leafIdx];

        IBox haloBox = makeHaloBox(octree.codeStart(leafIdx + octree.nInternalNodes()),
                                   octree.codeEnd(leafIdx + octree.nInternalNodes()), radius, globalBox);
        findCollisions(octree, collisions[leafIdx], haloBox);
    }

    return collisions;
}

} // namespace cstone