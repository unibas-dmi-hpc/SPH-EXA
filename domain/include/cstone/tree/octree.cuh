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

#include <thrust/device_vector.h>

#include "cstone/cuda/errorcheck.cuh"
#include "cstone/util/util.hpp"
#include "octree.hpp"

namespace cstone
{

//! @brief see computeNodeCounts
template<class KeyType>
__global__ void computeNodeCountsKernel(const KeyType* tree, unsigned* counts, TreeNodeIndex nNodes, const KeyType* codesStart,
                                        const KeyType* codesEnd, unsigned maxCount)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < nNodes)
    {
        counts[tid] = calculateNodeCount(tree, tid, codesStart, codesEnd, maxCount);
    }
}

//! @brief see updateNodeCounts
template<class KeyType>
__global__ void updateNodeCountsKernel(const KeyType* tree, unsigned* counts, TreeNodeIndex nNodes, const KeyType* codesStart,
                                       const KeyType* codesEnd, unsigned maxCount)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < nNodes)
    {
        unsigned firstGuess  = counts[tid];
        unsigned secondGuess = counts[min(tid+1, nNodes-1)];

        //unsigned secondGuess = __shfl_down_sync(0xffffffff, firstGuess, 1);
        counts[tid] = updateNodeCount(tid, tree, firstGuess, secondGuess,
                                      codesStart, codesEnd, maxCount);
    }
}

//! @brief used to communicate required node search range for computeNodeCountsKernel back to host
__device__ TreeNodeIndex populatedNodes[2];

//! @brief compute first and last non-empty nodes in the tree
template<class KeyType>
__global__ void findPopulatedNodes(const KeyType* tree, TreeNodeIndex nNodes, const KeyType* codesStart, const KeyType* codesEnd)
{
    if (threadIdx.x == 0 && codesStart != codesEnd)
    {
        populatedNodes[0] = stl::upper_bound(tree, tree + nNodes, *codesStart) - tree - 1;
        populatedNodes[1] = stl::upper_bound(tree, tree + nNodes, *(codesEnd - 1)) - tree;
    }
}

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
void computeNodeCountsGpu(const KeyType* tree, unsigned* counts, TreeNodeIndex nNodes, const KeyType* codesStart,
                          const KeyType* codesEnd, unsigned maxCount, bool useCountsAsGuess = false)
{
    TreeNodeIndex popNodes[2];

    findPopulatedNodes<<<1,1>>>(tree, nNodes, codesStart, codesEnd);
    cudaMemcpyFromSymbol(popNodes, populatedNodes, 2 * sizeof(TreeNodeIndex));

    cudaMemset(counts, 0, popNodes[0] * sizeof(unsigned));
    cudaMemset(counts + popNodes[1], 0, (nNodes - popNodes[1]) * sizeof(unsigned));

    constexpr unsigned nThreads = 256;
    if (useCountsAsGuess)
    {
        thrust::exclusive_scan(thrust::device, counts + popNodes[0], counts + popNodes[1], counts + popNodes[0]);
        updateNodeCountsKernel<<<iceil(popNodes[1] - popNodes[0], nThreads), nThreads>>>
            (tree + popNodes[0], counts + popNodes[0], popNodes[1] - popNodes[0], codesStart, codesEnd, maxCount);
    }
    else
    {
        computeNodeCountsKernel<<<iceil(popNodes[1] - popNodes[0], nThreads), nThreads>>>
            (tree + popNodes[0], counts + popNodes[0], popNodes[1] - popNodes[0], codesStart, codesEnd, maxCount);
    }
}

//! @brief this symbol is used to keep track of octree structure changes and detect convergence
__device__ int rebalanceChangeCounter;

/*! @brief Compute split or fuse decision for each octree node in parallel
 *
 * @tparam KeyType         32- or 64-bit unsigned integer type
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
template<class KeyType>
__global__ void rebalanceDecisionKernel(const KeyType* tree, const unsigned* counts, TreeNodeIndex nNodes,
                                        unsigned bucketSize, TreeNodeIndex* nodeOps)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < nNodes)
    {
        int decision = calculateNodeOp(tree, tid, counts, bucketSize);
        if (decision != 1) { rebalanceChangeCounter = 1;}
        nodeOps[tid] = decision;
    }
}

//! @brief construct new nodes in the balanced tree
template<class KeyType>
__global__ void processNodes(const KeyType* oldTree, const TreeNodeIndex* nodeOps,
                             TreeNodeIndex nOldNodes, TreeNodeIndex nNewNodes,
                             KeyType* newTree)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < nOldNodes)
    {
        processNode(tid, oldTree, nodeOps, newTree);
    }
    if (tid == nNewNodes)
    {
        newTree[tid] = nodeRange<KeyType>(0);
    }
}

__global__ void resetRebalanceCounter()
{
    rebalanceChangeCounter = 0;
}

/*! @brief split or fuse octree nodes based on node counts relative to bucketSize
 *
 * @tparam KeyType         32- or 64-bit unsigned integer type
 * @param[inout] tree      vector of octree nodes in cornerstone format
 *                         needs to satisfy the octree invariants
 * @param[in] counts       output particle counts per node, length = @p tree.size() - 1
 * @param[in] bucketSize   maximum particle count per (leaf) node and
 *                         minimum particle count (strictly >) for (implicit) internal nodes
 * @param      tmpTree     memory buffer for temporary use, neither input nor output
 * @param      workArray   memory buffer for temporary use, neither input nor output
 * @return                 true if converged, false otherwise
 */
template<class SfcVector>
bool rebalanceTreeGpu(SfcVector& tree, const unsigned* counts, unsigned bucketSize,
                      SfcVector& tmpTree, thrust::device_vector<TreeNodeIndex>& workArray)
{
    using KeyType = typename SfcVector::value_type;
    TreeNodeIndex nOldNodes = nNodes(tree);

    // +1 to store the total sum of the exclusive scan in the last element
    workArray.resize(tree.size());

    resetRebalanceCounter<<<1,1>>>();

    constexpr unsigned nThreads = 512;
    rebalanceDecisionKernel<<<iceil(nOldNodes, nThreads), nThreads>>>(
        thrust::raw_pointer_cast(tree.data()), counts, nOldNodes, bucketSize,
        thrust::raw_pointer_cast(workArray.data()));

    thrust::exclusive_scan(thrust::device, thrust::raw_pointer_cast(workArray.data()),
                          thrust::raw_pointer_cast(workArray.data()) + workArray.size(),
                          thrust::raw_pointer_cast(workArray.data()));

    // +1 for the end marker (nodeRange<KeyType>(0))
    tmpTree.resize(*workArray.rbegin() + 1);

    TreeNodeIndex nElements = stl::max(tree.size(), tmpTree.size());
    processNodes<<<iceil(nElements, nThreads), nThreads>>>(thrust::raw_pointer_cast(tree.data()),
                                                           thrust::raw_pointer_cast(workArray.data()), nOldNodes, nNodes(tmpTree),
                                                           thrust::raw_pointer_cast(tmpTree.data()));
    int changeCounter;
    cudaMemcpyFromSymbol(&changeCounter, rebalanceChangeCounter, sizeof(int));

    swap(tree, tmpTree);

    return changeCounter == 0;
}

/*! @brief update the octree with a single rebalance/count step
 *
 * @tparam KeyType           32- or 64-bit unsigned integer for morton code
 * @param[in]    codesStart  local particle Morton codes start
 * @param[in]    codesEnd    local particle morton codes end
 * @param[in]    bucketSize  maximum number of particles per node
 * @param[inout] tree        the octree leaf nodes (cornerstone format)
 * @param[inout] counts      the octree leaf node particle count
 * @param[-]     tmpTree     temporary array, will be resized as needed
 * @param[-]     workArray   temporary array, will be resized as needed
 * @param[in]    maxCount    if actual node counts are higher, they will be capped to @p maxCount
 * @return                   true if converged, false otherwise
 */
template<class KeyType>
bool updateOctreeGpu(const KeyType* codesStart, const KeyType* codesEnd, unsigned bucketSize,
                     thrust::device_vector<KeyType>& tree, thrust::device_vector<unsigned>& counts,
                     thrust::device_vector<KeyType>& tmpTree, thrust::device_vector<TreeNodeIndex>& workArray,
                     unsigned maxCount = std::numeric_limits<unsigned>::max())
{
    bool converged = rebalanceTreeGpu(tree, thrust::raw_pointer_cast(counts.data()), bucketSize, tmpTree, workArray);
    counts.resize(nNodes(tree));

    // local node counts
    computeNodeCountsGpu(thrust::raw_pointer_cast(tree.data()), thrust::raw_pointer_cast(counts.data()),
                         nNodes(tree), codesStart, codesEnd, maxCount, true);

    return converged;
}

} // namespace cstone
