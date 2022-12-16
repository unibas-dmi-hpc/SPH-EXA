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
 * @brief Naive collision detection implementation for validation and testing
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/traversal/boxoverlap.hpp"
#include "cstone/tree/cs_util.hpp"

namespace cstone
{
/*! @brief to-all implementation of findCollisions
 *
 * @tparam     KeyType        32- or 64-bit unsigned integer
 * @param[in]  tree           octree leaf nodes in cornerstone format
 * @param[out] collisionList  output list of indices of colliding nodes
 * @param[in]  collisionBox   query box to look for collisions
 *                            with leaf nodes
 *
 * Naive implementation without tree traversal for reference
 * and testing purposes
 */
template<class KeyType>
void findCollisions2All(gsl::span<const KeyType> tree,
                        std::vector<TreeNodeIndex>& collisionList,
                        const IBox& collisionBox)
{
    for (TreeNodeIndex idx = 0; idx < TreeNodeIndex(nNodes(tree)); ++idx)
    {
        IBox nodeBox = sfcIBox(sfcKey(tree[idx]), sfcKey(tree[idx + 1]));
        if (overlap<KeyType>(nodeBox, collisionBox)) { collisionList.push_back(idx); }
    }
}

//! @brief all-to-all implementation of findAllCollisions
template<class KeyType, class T>
std::vector<std::vector<TreeNodeIndex>>
findCollisionsAll2all(gsl::span<const KeyType> tree, const std::vector<T>& haloRadii, const Box<T>& globalBox)
{
    std::vector<std::vector<TreeNodeIndex>> collisions(tree.size() - 1);

    for (TreeNodeIndex leafIdx = 0; leafIdx < TreeNodeIndex(nNodes(tree)); ++leafIdx)
    {
        T radius = haloRadii[leafIdx];

        IBox haloBox = makeHaloBox(tree[leafIdx], tree[leafIdx + 1], radius, globalBox);
        findCollisions2All<KeyType>(tree, collisions[leafIdx], haloBox);
    }

    return collisions;
}

} // namespace cstone