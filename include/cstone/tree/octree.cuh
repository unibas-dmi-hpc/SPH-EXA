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

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <tuple>

#include <cuda.h>

#include <thrust/device_vector.h>
#include <thrust/scan.h>

#include "cstone/util.hpp"
#include "octree.hpp"

namespace cstone
{

/*! @brief count number of particles in each octree node
 *
 * @tparam I                32- or 64-bit unsigned integer type
 * @param[in]  tree         octree nodes given as Morton codes of length @a nNodes+1
 *                          needs to satisfy the octree invariants
 * @param[out] counts       output particle counts per node, length = @a nNodes
 * @param[in]  nNodes       number of nodes in tree
 * @param[in]  codesStart   sorted particle SFC code range start
 * @param[in]  codesEnd     sorted particle SFC code range end
 * @param[in]  maxCount     maximum particle count per node to store, this is used
 *                          to prevent overflow in MPI_Allreduce
 */
template<class I>
__global__ void computeNodeCountsKernel(const I* tree, unsigned* counts, TreeNodeIndex nNodes, const I* codesStart,
                                        const I* codesEnd, unsigned maxCount = std::numeric_limits<unsigned>::max())
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < nNodes)
    {
        counts[tid] = calculateNodeCount(tree, tid, codesStart, codesEnd, maxCount);
    }
}

/*! @brief Compute split or fuse decision for each octree node in parallel
 *
 * @tparam I               32- or 64-bit unsigned integer type
 * @param[in] tree         octree nodes given as Morton codes of length @a nNodes
 *                         needs to satisfy the octree invariants
 * @param[in] counts       output particle counts per node, length = @a nNodes
 * @param[in] nNodes       number of nodes in tree
 * @param[in] bucketSize   maximum particle count per (leaf) node and
 *                         minimum particle count (strictly >) for (implicit) internal nodes
 * @param[out] nodeOps     stores rebalance decision result for each node, length = @a nNodes
 * @param[out] converged   stores 0 upon return if converged, a non-zero positive integer otherwise.
 *                         The storage location is accessed concurrently and cuda-memcheck might detect
 *                         a data race, but this is irrelevant for correctness.
 *
 * For each node i in the tree, in nodeOps[i], stores
 *  - 0 if to be merged
 *  - 1 if unchanged,
 *  - 8 if to be split.
 */
template<class I>
__global__ void rebalanceDecisionKernel(const I* tree, const unsigned* counts, TreeNodeIndex nNodes,
                                        unsigned bucketSize, TreeNodeIndex* nodeOps, int* converged)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < nNodes)
    {
        nodeOps[tid] = calculateNodeOp(tree, tid, counts, bucketSize, converged);
    }
}

//! @brief construct new nodes in the balanced tree
template<class I>
__global__ void processNodes(const I* oldTree, const TreeNodeIndex* nodeOps,
                             TreeNodeIndex nOldNodes, TreeNodeIndex nNewNodes,
                             I* newTree)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < nOldNodes)
    {
        processNode(tid, oldTree, nodeOps, newTree);
    }
    if (tid == nNewNodes)
    {
        newTree[tid] = nodeRange<I>(0);
    }
}

/*! @brief split or fuse octree nodes based on node counts relative to bucketSize
 *
 * @tparam I               32- or 64-bit unsigned integer type
 * @param[in] tree         octree nodes given as Morton codes of length @a nNodes
 *                         needs to satisfy the octree invariants
 * @param[in] counts       output particle counts per node, length = @a nNodes
 * @param[in] nOldNodes       number of nodes in tree
 * @param[in] bucketSize   maximum particle count per (leaf) node and
 *                         minimum particle count (strictly >) for (implicit) internal nodes
 * @param[out] converged   optional boolean flag to indicate convergence
 * @return                 the rebalanced Morton code octree
 */
template<class SfcVector>
void rebalanceTreeGpu(SfcVector& tree, const unsigned* counts, TreeNodeIndex nOldNodes,
                      unsigned bucketSize, bool* converged = nullptr)
{
    using I = typename SfcVector::value_type;
    // +1 to store the total sum of the exclusive scan in the last element
    thrust::device_vector<TreeNodeIndex> nodeOps(nOldNodes + 1);
    thrust::device_vector<TreeNodeIndex> nodeOffsets(nOldNodes + 1);

    thrust::device_vector<int> changeCounter(1,0);
    constexpr unsigned nThreads = 512;
    rebalanceDecisionKernel<<<iceil(nOldNodes, nThreads), nThreads>>>(
        thrust::raw_pointer_cast(tree.data()), counts, nOldNodes, bucketSize,
        thrust::raw_pointer_cast(nodeOps.data()),
        thrust::raw_pointer_cast(changeCounter.data()));

    thrust::exclusive_scan(thrust::device, nodeOps.begin(), nodeOps.end(), nodeOffsets.begin());

    // +1 for the end marker (nodeRange<I>(0))
    SfcVector balancedTree(*nodeOffsets.rbegin() + 1);

    TreeNodeIndex nElements = stl::max(tree.size(), balancedTree.size());
    processNodes<<<iceil(nElements, nThreads), nThreads>>>(thrust::raw_pointer_cast(tree.data()),
                                                           thrust::raw_pointer_cast(nodeOffsets.data()), nOldNodes, nNodes(balancedTree),
                                                           thrust::raw_pointer_cast(balancedTree.data()));
    if (converged != nullptr)
    {
        *converged = (changeCounter[0] == 0);
    }

    swap(tree, balancedTree);
}

} // namespace cstone
