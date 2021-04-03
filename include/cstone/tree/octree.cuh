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
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include "cstone/util.hpp"
#include "octree.hpp"

namespace cstone
{

//! @brief see computeNodeCounts
template<class I>
__global__ void computeNodeCountsKernel(const I* tree, unsigned* counts, TreeNodeIndex nNodes, const I* codesStart,
                                        const I* codesEnd, unsigned maxCount)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < nNodes)
    {
        counts[tid] = calculateNodeCount(tree, tid, codesStart, codesEnd, maxCount);
    }
}

//! @brief see updateNodeCounts
template<class I>
__global__ void updateNodeCountsKernel(const I* tree, unsigned* counts, TreeNodeIndex nNodes, const I* codesStart,
                                       const I* codesEnd, unsigned maxCount)
{
    extern __shared__ unsigned guesses[];

    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < nNodes)
    {
        unsigned firstGuess  = counts[tid];

        guesses[threadIdx.x] = firstGuess;
        unsigned secondGuess = firstGuess;
        __syncthreads();
        if (threadIdx.x < blockDim.x - 1)
            secondGuess = guesses[threadIdx.x+1];

        //unsigned secondGuess = __shfl_down_sync(0xffffffff, firstGuess, 1);
        counts[tid] = updateNodeCount(tid, tree, firstGuess, secondGuess,
                                      codesStart, codesEnd, maxCount);
    }
}

//! @brief used to communicate required node search range for computeNodeCountsKernel back to host
__device__ TreeNodeIndex populatedNodes[2];

//! @brief compute first and last non-empty nodes in the tree
template<class I>
__global__ void findPopulatedNodes(const I* tree, TreeNodeIndex nNodes, const I* codesStart, const I* codesEnd)
{
    if (threadIdx.x == 0 && codesStart != codesEnd)
    {
        populatedNodes[0] = stl::upper_bound(tree, tree + nNodes, *codesStart) - tree - 1;
        populatedNodes[1] = stl::upper_bound(tree, tree + nNodes, *(codesEnd - 1)) - tree;
    }
}

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
void computeNodeCountsGpu(const I* tree, unsigned* counts, TreeNodeIndex nNodes, const I* codesStart,
                          const I* codesEnd, unsigned maxCount, bool useCountsAsGuess = false)
{
    TreeNodeIndex popNodes[2];

    findPopulatedNodes<<<1,1>>>(tree, nNodes, codesStart, codesEnd);
    cudaMemcpyFromSymbol(popNodes, populatedNodes, 2 * sizeof(TreeNodeIndex));

    cudaMemset(counts, 0, popNodes[0] * sizeof(unsigned));
    cudaMemset(counts + popNodes[1], 0, (nNodes - popNodes[1]) * sizeof(unsigned));

    constexpr unsigned nThreads = 256;
    if (useCountsAsGuess)
    {
        thrust::exclusive_scan(thrust::device, counts + popNodes[0], counts + popNodes[1], counts + popNodes[0], 0);
        updateNodeCountsKernel<<<iceil(popNodes[1] - popNodes[0], nThreads), nThreads, sizeof(unsigned) * nThreads>>>
            (tree + popNodes[0], counts + popNodes[0], popNodes[1] - popNodes[0], codesStart, codesEnd, maxCount);
    }
    else
    {
        computeNodeCountsKernel<<<iceil(popNodes[1] - popNodes[0], nThreads), nThreads>>>
            (tree + popNodes[0], counts + popNodes[0], popNodes[1] - popNodes[0], codesStart, codesEnd, maxCount);
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

/*! @brief compute an octree from morton codes for a specified bucket size

 * @tparam I               32- or 64-bit unsigned integer type
 * @param[in]    codesStart   particle morton code sequence start
 * @param[in]    codesEnd     particle morton code sequence end
 * @param[in]    bucketSize   maximum number of particles/codes per octree leaf node
 * @param[inout] tree         input tree for initial guess and converged output tree
 * @param[out]   counts       particle counts per node in @p tree
 * @param[in]    maxCount     if actual node counts are higher, they will be capped to @p maxCount
 *
 * See CPU version for an explanation about @p maxCount
 */
template<class SfcVector, class CountsVector, class I, class Reduce = void>
void computeOctreeGpu(const I* codesStart, const I* codesEnd, unsigned bucketSize,
                      SfcVector& tree, CountsVector& counts,
                      unsigned maxCount = std::numeric_limits<unsigned>::max())
{
    static_assert(std::is_same_v<typename SfcVector::value_type, I>);
    static_assert(std::is_same_v<typename CountsVector::value_type, unsigned>);

    if (!tree.size())
    {
        // tree containing just the root node
        tree.push_back(0);
        tree.push_back(nodeRange<I>(0));
    }

    bool converged = false;
    while (!converged)
    {
        counts.resize(nNodes(tree));
        computeNodeCountsGpu(thrust::raw_pointer_cast(tree.data()), thrust::raw_pointer_cast(counts.data()),
                             nNodes(tree), codesStart, codesEnd, maxCount);

        if constexpr (!std::is_same_v<void, Reduce>)
        {
            (void)Reduce{}(counts); // void cast to silence "warning: expression has no effect" from nvcc
        }

        rebalanceTreeGpu(tree, thrust::raw_pointer_cast(counts.data()), nNodes(tree), bucketSize, &converged);
    }
}

/*! @brief update the octree with a single rebalance/count step
 *
 * @tparam I                 32- or 64-bit unsigned integer for morton code
 * @tparam Reduce            functor for global counts reduction in distributed builds
 * @param[in]    codesStart  local particle Morton codes start
 * @param[in]    codesEnd    local particle morton codes end
 * @param[in]    bucketSize  maximum number of particles per node
 * @param[inout] tree        the octree leaf nodes (cornerstone format)
 * @param[inout] counts      the octree leaf node particle count
 * @param[in]    maxCount    if actual node counts are higher, they will be capped to @p maxCount
 */
template<class I, class Reduce = void>
void updateOctreeGpu(const I* codesStart, const I* codesEnd, unsigned bucketSize,
                     thrust::device_vector<I>& tree, thrust::device_vector<unsigned>& counts,
                     unsigned maxCount = std::numeric_limits<unsigned>::max())
{
    rebalanceTreeGpu(tree, thrust::raw_pointer_cast(counts.data()), nNodes(tree), bucketSize);
    counts.resize(nNodes(tree));

    // local node counts
    computeNodeCountsGpu(thrust::raw_pointer_cast(tree.data()), thrust::raw_pointer_cast(counts.data()),
                         nNodes(tree), codesStart, codesEnd, maxCount, true);
    // global node count sums when using distributed builds
    if constexpr (!std::is_same_v<void, Reduce>) (void)Reduce{}(counts);
}

} // namespace cstone
