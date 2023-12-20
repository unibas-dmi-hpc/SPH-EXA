/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
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
 * @brief SFC key injection into cornerstone arrays to enforce the presence of specified keys
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/tree/csarray.hpp"
#include "cstone/tree/csarray_gpu.h"
#include "cstone/tree/octree.hpp"
#include "cstone/primitives/primitives_gpu.h"

namespace cstone
{

/*! @brief inject specified keys into a cornerstone leaf tree
 *
 * @tparam KeyType    32- or 64-bit unsigned integer
 * @param[inout] tree   cornerstone octree
 * @param[in]    keys   list of SFC keys to insert
 *
 * This function needs to insert more than just @p keys, due the cornerstone
 * invariant of consecutive nodes always having a power-of-8 difference.
 * This means that each subdividing a node, all 8 children always have to be added.
 */
template<class KeyType, class Alloc>
void injectKeys(std::vector<KeyType, Alloc>& tree, gsl::span<const KeyType> keys)
{
    std::vector<KeyType> spanningKeys(keys.begin(), keys.end());
    spanningKeys.push_back(0);
    spanningKeys.push_back(nodeRange<KeyType>(0));
    std::sort(begin(spanningKeys), end(spanningKeys));
    auto uit = std::unique(begin(spanningKeys), end(spanningKeys));
    spanningKeys.erase(uit, end(spanningKeys));

    // spanningTree is a list of all the missing nodes needed to resolve the mandatory keys
    auto spanningTree = computeSpanningTree<KeyType>(spanningKeys);
    tree.reserve(tree.size() + spanningTree.size());

    // spanningTree is now inserted into newLeaves
    std::copy(begin(spanningTree), end(spanningTree), std::back_inserter(tree));

    // cleanup, restore invariants: sorted-ness, no-duplicates
    std::sort(begin(tree), end(tree));
    uit = std::unique(begin(tree), end(tree));
    tree.erase(uit, end(tree));
}

template<class Otree, class DevVec>
void injectKeysGpu(Otree& tree, DevVec& leaves, const DevVec& keys)
{
    reallocate(leaves, leaves.size() + keys.size(), 1.0);
    memcpyD2D(rawPtr(keys), keys.size(), rawPtr(leaves) + leaves.size() - keys.size());

    reallocateDestructive(tree.prefixes, leaves.size(), 1.0);
    sortGpu(rawPtr(leaves), rawPtr(leaves) + leaves.size(), rawPtr(tree.prefixes));

    reallocateDestructive(tree.childOffsets, nNodes(leaves), 1.0);
    TreeNodeIndex* spanOps = rawPtr(tree.childOffsets);
    reallocateDestructive(tree.internalToLeaf, leaves.size(), 1.0);
    TreeNodeIndex* spanOpsScan = rawPtr(tree.internalToLeaf);

    countSfcGapsGpu(rawPtr(leaves), nNodes(leaves), spanOps);
    exclusiveScanGpu(spanOps, spanOps + leaves.size(), spanOpsScan);

    TreeNodeIndex numNodesGap;
    memcpyD2H(spanOpsScan + leaves.size() - 1, 1, &numNodesGap);

    reallocateDestructive(tree.prefixes, numNodesGap + 1, 1.0);
    fillSfcGapsGpu(rawPtr(leaves), nNodes(leaves), spanOpsScan, rawPtr(tree.prefixes));
    swap(leaves, tree.prefixes);
}

} // namespace cstone
