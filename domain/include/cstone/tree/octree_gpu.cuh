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

#include "cstone/cuda/cuda_utils.cuh"
#include "octree.hpp"

namespace cstone
{

/*! @brief count number of particles in each octree node
 *
 * @tparam KeyType          32- or 64-bit unsigned integer type
 * @param[in]  tree         octree nodes given as Morton codes of length @a nNodes+1
 *                          needs to satisfy the octree invariants
 * @param[out] counts       output particle counts per node, length = @a nNodes
 * @param[in]  nNodes       number of nodes in tree
 * @param[in]  codesStart   sorted particle SFC code range start
 * @param[in]  codesEnd     sorted particle SFC code range end
 * @param[in]  maxCount     maximum particle count per node to store, this is used
 *                          to prevent overflow in MPI_Allreduce
 */
template<class KeyType>
extern void computeNodeCountsGpu(const KeyType* tree,
                                 unsigned* counts,
                                 TreeNodeIndex nNodes,
                                 const KeyType* codesStart,
                                 const KeyType* codesEnd,
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

/*! @brief update the octree with a single rebalance/count step
 *
 * @tparam KeyType           32- or 64-bit unsigned integer for morton code
 * @param[in]    firstKey    first local particle SFC key
 * @param[in]    lastKey     last local particle SFC key
 * @param[in]    bucketSize  maximum number of particles per node
 * @param[inout] tree        the octree leaf nodes (cornerstone format)
 * @param[inout] counts      the octree leaf node particle count
 * @param[-]     tmpTree     temporary array, will be resized as needed
 * @param[-]     workArray   temporary array, will be resized as needed
 * @param[in]    maxCount    if actual node counts are higher, they will be capped to @p maxCount
 * @return                   true if converged, false otherwise
 */
template<class KeyType, class DevKeyVec, class DevCountVec, class DevIdxVec>
bool updateOctreeGpu(const KeyType* firstKey,
                     const KeyType* lastKey,
                     unsigned bucketSize,
                     DevKeyVec& tree,
                     DevCountVec& counts,
                     DevKeyVec& tmpTree,
                     DevIdxVec& workArray,
                     unsigned maxCount = std::numeric_limits<unsigned>::max())
{
    workArray.resize(tree.size());
    TreeNodeIndex newNumNodes =
        computeNodeOpsGpu(rawPtr(tree), nNodes(tree), rawPtr(counts), bucketSize, rawPtr(workArray));

    tmpTree.resize(newNumNodes + 1);
    bool converged = rebalanceTreeGpu(rawPtr(tree), nNodes(tree), newNumNodes, rawPtr(workArray), rawPtr(tmpTree));

    swap(tree, tmpTree);
    counts.resize(nNodes(tree));

    // local node counts
    computeNodeCountsGpu(rawPtr(tree), rawPtr(counts), nNodes(tree), firstKey, lastKey, maxCount, true);

    return converged;
}

} // namespace cstone
