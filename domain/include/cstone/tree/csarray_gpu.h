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
 * @brief Generation of local and global octrees in cornerstone format on the GPU
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * See octree.hpp for a description of the cornerstone format.
 */

#pragma once

#include <limits>

#include "csarray.hpp"

namespace cstone
{

/*! @brief count number of particles in each octree node
 *
 * @tparam KeyType          32- or 64-bit unsigned integer type
 * @param[in]  tree         octree nodes given as Morton codes of length @a nNodes+1
 *                          needs to satisfy the octree invariants
 * @param[out] counts       output particle counts per node, length = @a nNodes
 * @param[in]  numNodes       number of nodes in tree
 * @param[in]  firstKey   sorted particle SFC code range start
 * @param[in]  lastKey     sorted particle SFC code range end
 * @param[in]  maxCount     maximum particle count per node to store, this is used
 *                          to prevent overflow in MPI_Allreduce
 */
template<class KeyType>
extern void computeNodeCountsGpu(const KeyType* tree,
                                 unsigned* counts,
                                 TreeNodeIndex numNodes,
                                 const KeyType* firstKey,
                                 const KeyType* lastKey,
                                 unsigned maxCount,
                                 bool useCountsAsGuess = false);

/*! @brief split or fuse octree nodes based on node counts relative to bucketSize
 *
 * @tparam KeyType         32- or 64-bit unsigned integer type
 * @param[in] tree         vector of octree nodes in cornerstone format, length = @p numNodes + 1
 * @param[in] numNodes     number of nodes in @p tree
 * @param[in] counts       output particle counts per node, length = @p tree.size() - 1
 * @param[in] bucketSize   maximum particle count per (leaf) node and
 * @param[out] nodeOps     node transformation codes, length = @p numNodes + 1
 * @return                 number of nodes in the future rebalanced tree
 */
template<class KeyType>
extern TreeNodeIndex computeNodeOpsGpu(
    const KeyType* tree, TreeNodeIndex numNodes, const unsigned* counts, unsigned bucketSize, TreeNodeIndex* nodeOps);

template<class KeyType>
extern bool rebalanceTreeGpu(const KeyType* tree,
                             TreeNodeIndex numNodes,
                             TreeNodeIndex newNumNodes,
                             const TreeNodeIndex* nodeOps,
                             KeyType* newTree);

} // namespace cstone
