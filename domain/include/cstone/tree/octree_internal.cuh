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

#include "octree_internal.hpp"
#include "cstone/tree/btree.cuh"

namespace cstone {

//! @brief see nodeDepth, note: depths must be initialized to zero, as in the CPU version
template<class I>
__global__ void nodeDepthKernel(const OctreeNode<I>* octree, TreeNodeIndex nNodes, TreeNodeIndex* depths)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < nNodes)
    {
        nodeDepthElement(tid, octree, depths);
    }
}

//! Octree GPU data view for use in kernel code
template<class KeyType>
struct OctreeGpuDataView
{

    TreeNodeIndex numLeafNodes;
    TreeNodeIndex numInternalNodes;

    BinaryNode<KeyType>* binaryTree;

    KeyType*       prefixes;
    TreeNodeIndex* binaryToOct;
    TreeNodeIndex* octToBinary;
    TreeNodeIndex* nodeOrder;
    TreeNodeIndex* inverseNodeOrder;
    TreeNodeIndex* childOffsets;
    TreeNodeIndex* parents;
    TreeNodeIndex* levelRange;
};

/*! @brief determine which binary nodes correspond to octree nodes
 *
 * @tparam KeyType         unsigned 32- or 64-bit integer
 * @param[in]  binaryTree  binary radix tree nodes, length @p numNodes
 * @param[in]  numNodes    number of binary radix tree nodes
 * @param[out] binaryToOct for each binary node, store 1 if prefix bit length is divisible by 3
 */
template<class KeyType>
__global__ void
enumeratePrefixes(const BinaryNode<KeyType>* binaryTree, TreeNodeIndex numNodes, TreeNodeIndex* binaryToOct)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < numNodes)
    {
        int  prefixLength = decodePrefixLength(binaryTree[tid].prefix);
        bool divisibleBy3 = prefixLength % 3 == 0;
        binaryToOct[tid] = (divisibleBy3) ? 1 : 0;
    }
}

/*! @brief map octree nodes back to binary nodes
 *
 * @param[in]  binaryToOct     translation map from binary to octree nodes, length @p numBinaryNodes
 * @param[in]  numBinaryNodes  number of binary tree nodes
 * @param[out] octToBinary     the inversion of binaryToOct, octToBinary[binaryToOct[i]] == i
 */
__global__ void translateToOct(const TreeNodeIndex* binaryToOct, int numBinaryNodes, TreeNodeIndex* octToBinary)
{
    int tid = int(blockDim.x * blockIdx.x + threadIdx.x);
    if (tid < numBinaryNodes)
    {
        bool isOctreeNode = (binaryToOct[tid+1] - binaryToOct[tid]) == 1;
        if (isOctreeNode)
        {
            int octreeNodeIndex = binaryToOct[tid];
            octToBinary[octreeNodeIndex] = tid;
        }
    }
}

/*! @brief combine internal and leaf tree parts into single arrays in the binary tree order
 *
 * @tparam KeyType               unsigned 32- or 64-bit integer
 * @param[in]  binaryTree        binary radix tree nodes, length numNodes - numInternalNodes - 1
 * @param[in]  numInternalNodes  number of internal (output) octree nodes
 * @param[in]  numNodes          total number of nodes
 * @param[in]  leaves            cornerstone SFC keys used to compute binaryTree, length numNodes - numInternalNodes + 1
 * @param[in]  octToBinary       translation map from octree to binary node indices
 * @param[out] prefixes          output octree SFC keys, length @p numNodes
 *                               NOTE: keys are prefixed with Warren-Salmon placeholder bits!
 * @param[out] nodeOrder         iota 0,1,2,3,... sequence for later use, length @p numNodes
 * @param[out] inverseNodeOrder  iota 0,1,2,3,... sequence for later use, length @p numNodes
 *
 * Unsorted binary radix tree ordering: first all internal nodes, then leaf nodes
 *
 *    binaryTree |---------------------------------------------------|
 *                       ^                      |
 *      octToBinary   |--|  |-------------------|  binaryToOct
 *                    |     V
 *    prefixes   |------------|--------------------------------|
 *    levels     |------------|--------------------------------|
 *                 internal        leaves
 */
template<class KeyType>
__global__ void createUnsortedLayout(const BinaryNode<KeyType>* binaryTree, int numInternalNodes, int numNodes,
                                     const KeyType* leaves, const TreeNodeIndex* octToBinary,
                                     KeyType* prefixes, TreeNodeIndex* nodeOrder, TreeNodeIndex* inverseNodeOrder)
{
    int tid = int(blockDim.x * blockIdx.x + threadIdx.x);
    // internal node
    if (tid < numInternalNodes)
    {
        TreeNodeIndex binaryIndex = octToBinary[tid];
        prefixes[tid]             = binaryTree[binaryIndex].prefix;

        nodeOrder[tid] = tid;
        inverseNodeOrder[tid] = tid;
    }
    // leaf node
    else if (tid < numNodes)
    {
        TreeNodeIndex leafIdx = tid - numInternalNodes;
        unsigned level        = treeLevel(leaves[leafIdx + 1] - leaves[leafIdx]);
        prefixes[tid]         = encodePlaceholderBit(leaves[leafIdx], 3 * level);

        nodeOrder[tid] = tid;
        inverseNodeOrder[tid] = tid;
    }
}

/*! @brief extract parent/child relationships from binary tree and translate to sorted order
 *
 * @tparam KeyType               unsigned 32- or 64-bit integer
 * @param[in]  binaryTree        binary radix tree nodes
 * @param[in]  numInternalNodes  number of internal octree nodes
 * @param[in]  octToBinary       internal octree to binary node index translation map, length @p numInternalNodes
 * @param[in]  binaryToOct       binary node to internal octree node index translation map
 * @param[in]  inverseNodeOrder  translation map from unsorted layout to level/SFC sorted octree layout
 *                               length is total number of octree nodes, internal + leaves
 * @param[out] childOffsets      octree node index of first child for each node, length is total number of nodes
 * @param[out] parents           parent index of for each node which is the first of 8 siblings
 *                               i.e. the parent of node i is stored at parents[(i - 1)/8]
 */
template<class KeyType>
__global__ void linkTree(const BinaryNode<KeyType>* binaryTree,
                         TreeNodeIndex numInternalNodes,
                         const TreeNodeIndex* octToBinary,
                         const TreeNodeIndex* binaryToOct,
                         const TreeNodeIndex* inverseNodeOrder,
                         TreeNodeIndex* childOffsets,
                         TreeNodeIndex* parents)
{
    // loop over octree nodes index in unsorted layout A
    unsigned idxA = blockDim.x * blockIdx.x + threadIdx.x;
    if (idxA < numInternalNodes)
    {
        TreeNodeIndex binaryIndex = octToBinary[idxA];
        TreeNodeIndex firstChild  = binaryTree[binaryTree[binaryTree[binaryIndex].child[0]].child[0]].child[0];

        // octree node index in sorted layout B
        TreeNodeIndex idxB = inverseNodeOrder[idxA];

        // child node index in unsorted layout A
        TreeNodeIndex childA =
            (isLeafIndex(firstChild)) ? loadLeafIndex(firstChild) + numInternalNodes : binaryToOct[firstChild];

        // node index in layout B
        TreeNodeIndex childB = inverseNodeOrder[childA];
        // an internal node must have a valid child
        assert(childB > 0);

        childOffsets[idxB]        = childB;
        // We only store the parent once for every group of 8 siblings. This works as long as each node always has
        // 8 siblings. Subtract one because the root has no siblings.
        parents[(childB - 1) / 8] = idxB;
    }
}

//! @brief determine the octree subdivision level boundaries
template<class KeyType>
__global__ void getLevelRange(const KeyType* nodeKeys, TreeNodeIndex numNodes, TreeNodeIndex* levelRange)
{
    int tid = int(blockDim.x * blockIdx.x + threadIdx.x);
    if (tid < numNodes - 1)
    {
        unsigned l1 = decodePrefixLength(nodeKeys[tid]);
        unsigned l2 = decodePrefixLength(nodeKeys[tid + 1]);

        if (l1 != l2) { levelRange[l2 / 3] = tid + 1; }
    }
    if (tid == numNodes - 1)
    {
        unsigned l1        = decodePrefixLength(nodeKeys[tid]);
        levelRange[l1 / 3 + 1] = tid + 1;
    }
}

/*! @brief functor to sort octree nodes first according to level, then by SFC key
 *
 * Note: takes SFC keys with Warren-Salmon placeholder bits in place as arguments
 */
template<class KeyType>
struct compareLevelThenPrefix
{
    HOST_DEVICE_FUN bool operator()(KeyType a, KeyType b) const
    {
        unsigned prefix_a = cstone::decodePrefixLength(a);
        unsigned prefix_b = cstone::decodePrefixLength(b);

        if (prefix_a < prefix_b)
        {
            return true;
        }
        else if (prefix_b < prefix_a)
        {
            return false;
        }
        else
        {
            return cstone::decodePlaceholderBit(a) < cstone::decodePlaceholderBit(b);
        }
    }
};

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

    createBinaryTreeGpu(cstoneTree, d.numLeafNodes, d.binaryTree);

    // we ignore the last binary tree node which is a duplicate root node
    TreeNodeIndex numBinaryNodes = d.numLeafNodes - 1;

    enumeratePrefixes<<<iceil(numBinaryNodes, numThreads), numThreads>>>(d.binaryTree, numBinaryNodes, d.binaryToOct);

    thrust::exclusive_scan(thrust::device, d.binaryToOct, d.binaryToOct + d.numLeafNodes, d.binaryToOct);

    translateToOct<<<iceil(numBinaryNodes, numThreads), numThreads>>>(d.binaryToOct, numBinaryNodes, d.octToBinary);

    TreeNodeIndex numNodes = d.numInternalNodes + d.numLeafNodes;
    createUnsortedLayout<<<iceil(numNodes, numThreads), numThreads>>>(d.binaryTree, d.numInternalNodes, numNodes,
                                                                      cstoneTree, d.octToBinary, d.prefixes,
                                                                      d.nodeOrder, d.inverseNodeOrder);

    thrust::sort_by_key(thrust::device, d.prefixes, d.prefixes + numNodes, d.nodeOrder, compareLevelThenPrefix<KeyType>{});
    // arrays now in sorted layout B

    // temporarily repurpose childOffsets as space for sort key
    thrust::copy(thrust::device, d.nodeOrder, d.nodeOrder + numNodes, d.childOffsets);
    thrust::sort_by_key(thrust::device, d.childOffsets, d.childOffsets + numNodes, d.inverseNodeOrder);
    thrust::fill(thrust::device, d.childOffsets, d.childOffsets + numNodes, 0);

    linkTree<<<iceil(d.numInternalNodes, numThreads), numThreads>>>(
        d.binaryTree, d.numInternalNodes, d.octToBinary, d.binaryToOct, d.inverseNodeOrder, d.childOffsets, d.parents);

    thrust::fill(thrust::device, d.levelRange, d.levelRange + maxTreeLevel<KeyType>{} + 2, 0);
    getLevelRange<<<iceil(numNodes, numThreads), numThreads>>>(d.prefixes, numNodes, d.levelRange);
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

        binaryTree.resize(numLeafNodes);

        prefixes.resize(numNodes);

        binaryToOct.resize(numLeafNodes);
        octToBinary.resize(numInternalNodes);

        childOffsets.resize(numNodes);
        nodeOrder.resize(numNodes);
        inverseNodeOrder.resize(numNodes);

        parents.resize((numNodes - 1) / 8);

        //+1 due to level 0 and +1 due to the upper bound for the last level
        levelRange.resize(maxTreeLevel<KeyType>{} + 2);
    }

    OctreeGpuDataView<KeyType> getData()
    {
        return {numLeafNodes,
                numInternalNodes,
                thrust::raw_pointer_cast(binaryTree.data()),
                thrust::raw_pointer_cast(prefixes.data()),
                thrust::raw_pointer_cast(binaryToOct.data()),
                thrust::raw_pointer_cast(octToBinary.data()),
                thrust::raw_pointer_cast(nodeOrder.data()),
                thrust::raw_pointer_cast(inverseNodeOrder.data()),
                thrust::raw_pointer_cast(childOffsets.data()),
                thrust::raw_pointer_cast(parents.data()),
                thrust::raw_pointer_cast(levelRange.data()),
        };
    }

    TreeNodeIndex numLeafNodes{0};
    TreeNodeIndex numInternalNodes{0};

    thrust::device_vector<BinaryNode<KeyType>> binaryTree;

    thrust::device_vector<KeyType>  prefixes;
    thrust::device_vector<TreeNodeIndex> binaryToOct;
    thrust::device_vector<TreeNodeIndex> octToBinary;
    thrust::device_vector<TreeNodeIndex> nodeOrder;
    thrust::device_vector<TreeNodeIndex> inverseNodeOrder;
    thrust::device_vector<TreeNodeIndex> childOffsets;
    thrust::device_vector<TreeNodeIndex> parents;
    thrust::device_vector<TreeNodeIndex> levelRange;
};

} // namespace cstone