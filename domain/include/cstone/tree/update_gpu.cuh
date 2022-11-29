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
#include "cstone/tree/csarray_gpu.h"
#include "csarray.hpp"

namespace cstone
{

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
