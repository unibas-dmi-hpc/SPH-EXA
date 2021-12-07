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

#pragma once

#include "cstone/cuda/annotation.hpp"

#include "cstone/util/array.hpp"

namespace cstone
{

//! @brief Controls the node index type, has to be signed. Change to 64-bit if more than 2 billion tree nodes are
//! required.
using TreeNodeIndex = int;

using LocalParticleIndex = unsigned;

template<class T>
using Vec3 = util::array<T, 3>;

using TestType = float;

//! @brief checks whether a binary tree index corresponds to a leaf index
HOST_DEVICE_FUN
constexpr bool isLeafIndex(TreeNodeIndex nodeIndex) { return nodeIndex < 0; }

//! @brief convert a leaf index to the storage format
HOST_DEVICE_FUN
constexpr TreeNodeIndex storeLeafIndex(TreeNodeIndex index)
{
    // -2^31 or -2^63
    constexpr auto offset = TreeNodeIndex(-(1ul << (8 * sizeof(TreeNodeIndex) - 1)));
    return index + offset;
}

//! @brief restore a leaf index from the storage format
HOST_DEVICE_FUN
constexpr TreeNodeIndex loadLeafIndex(TreeNodeIndex index)
{
    constexpr auto offset = TreeNodeIndex(-(1ul << (8 * sizeof(TreeNodeIndex) - 1)));
    return index - offset;
}

enum class P2pTags : int
{
    focusPeerCounts = 1000,
    haloRequestKeys = 2000,
    haloExchange    = 3000,
    domainExchange  = 4000
};

/*! @brief returns the number of nodes in a tree
 *
 * @tparam    Vector  a vector-like container that has a .size() member
 * @param[in] tree    input tree
 * @return            the number of nodes
 *
 * This makes it explicit that a vector of n Morton codes
 * corresponds to a tree with n-1 nodes.
 */
template<class Vector>
std::size_t nNodes(const Vector& tree)
{
    assert(tree.size());
    return tree.size() - 1;
}

} // namespace cstone
