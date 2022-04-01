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
 * @brief  Compute the internal part of a cornerstone octree on the GPU
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 */

#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cstone/tree/octree_internal.hpp"

namespace cstone {

//! Octree GPU data view for use in kernel code
template<class KeyType>
struct OctreeGpuDataView
{
    TreeNodeIndex numLeafNodes;
    TreeNodeIndex numInternalNodes;

    KeyType* prefixes;
    TreeNodeIndex* childOffsets;
    TreeNodeIndex* parents;
    TreeNodeIndex* levelRange;
    TreeNodeIndex* nodeOrder;
    TreeNodeIndex* inverseNodeOrder;
};

/*! @brief combine internal and leaf tree parts into a single array with the nodeKey prefixes
 *
 * @tparam     KeyType           unsigned 32- or 64-bit integer
 * @param[in]  leaves            cornerstone SFC keys, length numLeafNodes + 1
 * @param[in]  numInternalNodes  number of internal octree nodes
 * @param[in]  numLeafNodes      total number of nodes
 * @param[in]  binaryToOct       translation map from binary to octree nodes
 * @param[out] prefixes          output octree SFC keys, length @p numInternalNodes + numLeafNodes
 *                               NOTE: keys are prefixed with Warren-Salmon placeholder bits!
 * @param[out] nodeOrder         iota 0,1,2,3,... sequence for later use, length same as @p prefixes
 */
template<class KeyType>
__global__ void createUnsortedLayout(const KeyType* leaves,
                                     TreeNodeIndex numInternalNodes,
                                     TreeNodeIndex numLeafNodes,
                                     KeyType* prefixes,
                                     TreeNodeIndex* nodeOrder)
{
    int tid = int(blockDim.x * blockIdx.x + threadIdx.x);
    if (tid < numLeafNodes)
    {
        KeyType key                       = leaves[tid];
        unsigned level                    = treeLevel(leaves[tid + 1] - key);
        prefixes[tid + numInternalNodes]  = encodePlaceholderBit(key, 3 * level);
        nodeOrder[tid + numInternalNodes] = tid + numInternalNodes;

        unsigned prefixLength = commonPrefix(key, leaves[tid + 1]);
        if (prefixLength % 3 == 0 && tid < numLeafNodes - 1)
        {
            TreeNodeIndex octIndex = (tid + binaryKeyWeight(key, prefixLength / 3)) / 7;
            prefixes[octIndex]     = encodePlaceholderBit(key, prefixLength);
            nodeOrder[octIndex]    = octIndex;
        }
    }
}

/*! @brief extract parent/child relationships from binary tree and translate to sorted order
 *
 * @tparam     KeyType           unsigned 32- or 64-bit integer
 * @param[in]  prefixes          octree node prefixes in Warren-Salmon format
 * @param[in]  numInternalNodes  number of internal octree nodes
 * @param[in]  inverseNodeOrder  translation map from unsorted layout to level/SFC sorted octree layout
 *                               length is total number of octree nodes, internal + leaves
 * @param[in]  levelRange        indices of the first node at each level
 * @param[out] childOffsets      octree node index of first child for each node, length is total number of nodes
 * @param[out] parents           parent index of for each node which is the first of 8 siblings
 *                               i.e. the parent of node i is stored at parents[(i - 1)/8]
 */
template<class KeyType>
__global__ void linkTree(const KeyType* prefixes,
                         TreeNodeIndex numInternalNodes,
                         const TreeNodeIndex* inverseNodeOrder,
                         const TreeNodeIndex* levelRange,
                         TreeNodeIndex* childOffsets,
                         TreeNodeIndex* parents)
{
    // loop over octree nodes index in unsorted layout A
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < numInternalNodes)
    {
        TreeNodeIndex idxA    = inverseNodeOrder[tid];
        KeyType prefix        = prefixes[idxA];
        KeyType nodeKey       = decodePlaceholderBit(prefix);
        unsigned prefixLength = decodePrefixLength(prefix);
        unsigned level        = prefixLength / 3;
        assert(level < maxTreeLevel<KeyType>{});

        KeyType childPrefix = encodePlaceholderBit(nodeKey, prefixLength + 3);

        TreeNodeIndex leafSearchStart = levelRange[level + 1];
        TreeNodeIndex leafSearchEnd   = levelRange[level + 2];
        TreeNodeIndex childIdx =
            stl::lower_bound(prefixes + leafSearchStart, prefixes + leafSearchEnd, childPrefix) - prefixes;

        if (childIdx != leafSearchEnd && childPrefix == prefixes[childIdx])
        {
            childOffsets[idxA] = childIdx;
            // We only store the parent once for every group of 8 siblings.
            // This works as long as each node always has 8 siblings.
            // Subtract one because the root has no siblings.
            parents[(childIdx - 1) / 8] = idxA;
        }
    }
}

//! @brief determine the octree subdivision level boundaries
template<class KeyType>
__global__ void getLevelRange(const KeyType* nodeKeys, TreeNodeIndex numNodes, TreeNodeIndex* levelRange)
{
    unsigned level = blockIdx.x;
    auto it = stl::lower_bound(nodeKeys, nodeKeys + numNodes, encodePlaceholderBit(KeyType(0), 3 * level));
    levelRange[level] = TreeNodeIndex(it - nodeKeys);
}

//! @brief computes the inverse of the permutation given by @p order
__global__ void invertOrder(const TreeNodeIndex* order, TreeNodeIndex* inverseOrder, TreeNodeIndex numNodes)
{
    int tid = int(blockDim.x * blockIdx.x + threadIdx.x);
    if (tid < numNodes)
    {
        inverseOrder[order[tid]] = tid;
    }
}

/*! @brief construct the internal octree part of a given octree leaf cell array on the GPU
 *
 * @tparam       KeyType     unsigned 32- or 64-bit integer
 * @param[in]    cstoneTree  GPU buffer with the SFC leaf cell keys
 * @param[inout] d           input:  pointers to pre-allocated GPU buffers for octree cells
 *                           ouptut: fully linked octree
 *
 * This does not allocate memory on the GPU, (except thrust temp buffers for scans and sorting)
 */
template<class KeyType>
void buildInternalOctreeGpu(const KeyType* cstoneTree, OctreeGpuDataView<KeyType> d)
{
    constexpr unsigned numThreads = 256;

    TreeNodeIndex numNodes = d.numInternalNodes + d.numLeafNodes;
    createUnsortedLayout<<<iceil(numNodes, numThreads), numThreads>>>(cstoneTree, d.numInternalNodes, d.numLeafNodes,
                                                                      d.prefixes, d.nodeOrder);

    thrust::sort_by_key(thrust::device, d.prefixes, d.prefixes + numNodes, d.nodeOrder);

    invertOrder<<<iceil(numNodes, numThreads), numThreads>>>(d.nodeOrder, d.inverseNodeOrder, numNodes);
    getLevelRange<<<maxTreeLevel<KeyType>{} + 2, 1>>>(d.prefixes, numNodes, d.levelRange);

    thrust::fill(thrust::device, d.childOffsets, d.childOffsets + numNodes, 0);
    linkTree<<<iceil(d.numInternalNodes, numThreads), numThreads>>>(d.prefixes, d.numInternalNodes, d.inverseNodeOrder,
                                                                    d.levelRange, d.childOffsets, d.parents);
}

//! @brief provides a place to live for GPU resident octree data
template<class KeyType>
class OctreeGpuDataAnchor
{
public:
    void resize(TreeNodeIndex numCsLeafNodes)
    {
        numLeafNodes           = numCsLeafNodes;
        numInternalNodes       = (numLeafNodes - 1) / 7;
        TreeNodeIndex numNodes = numLeafNodes + numInternalNodes;

        prefixes.resize(numNodes);
        // +1 to accommodate nodeOffsets in FocusedOctreeCore::update when numNodes == 1
        childOffsets.resize(numNodes + 1);
        parents.resize((numNodes - 1) / 8);
        //+1 due to level 0 and +1 due to the upper bound for the last level
        levelRange.resize(maxTreeLevel<KeyType>{} + 2);

        nodeOrder.resize(numNodes);
        inverseNodeOrder.resize(numNodes);
    }

    OctreeGpuDataView<KeyType> getData()
    {
        return {numLeafNodes,
                numInternalNodes,
                thrust::raw_pointer_cast(prefixes.data()),
                thrust::raw_pointer_cast(childOffsets.data()),
                thrust::raw_pointer_cast(parents.data()),
                thrust::raw_pointer_cast(levelRange.data()),
                thrust::raw_pointer_cast(nodeOrder.data()),
                thrust::raw_pointer_cast(inverseNodeOrder.data())};
    }

    TreeNodeIndex numLeafNodes{0};
    TreeNodeIndex numInternalNodes{0};

    //! @brief the SFC key and level of each node (Warren-Salmon placeholder-bit), length = numNodes
    thrust::device_vector<KeyType> prefixes;
    //! @brief the index of the first child of each node, a value of 0 indicates a leaf, length = numNodes
    thrust::device_vector<TreeNodeIndex> childOffsets;
    //! @brief stores the parent index for every group of 8 sibling nodes, length the (numNodes - 1) / 8
    thrust::device_vector<TreeNodeIndex> parents;
    //! @brief store the first node index of every tree level, length = maxTreeLevel + 2
    thrust::device_vector<TreeNodeIndex> levelRange;

    //! @brief maps internal to leaf (cstone) order
    thrust::device_vector<TreeNodeIndex> nodeOrder;
    //! @brief maps leaf (cstone) order to internal level-sorted order
    thrust::device_vector<TreeNodeIndex> inverseNodeOrder;
};

} // namespace cstone
