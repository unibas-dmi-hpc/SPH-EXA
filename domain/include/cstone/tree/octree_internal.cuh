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
#include <thrust/scan.h>
#include <thrust/sort.h>

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

template<class KeyType>
struct OctreeGpuDataView
{

    TreeNodeIndex numLeafNodes{0};
    TreeNodeIndex numInternalNodes{0};

    BinaryNode<KeyType>* binaryTree;

    KeyType*       prefixes;
    unsigned*      levels;
    TreeNodeIndex* binaryToOct;
    TreeNodeIndex* octToBinary;
    TreeNodeIndex* nodeOrder;
    TreeNodeIndex* inverseNodeOrder;
    TreeNodeIndex* childOffsets;
    TreeNodeIndex* parents;
};

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

__global__ void translateToOct(const TreeNodeIndex* binaryToOct, int numBinaryNodes, TreeNodeIndex* octToBinary)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
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

template<class KeyType>
__global__ void createUnsortedLayout(const BinaryNode<KeyType>* binaryTree, int numInternalNodes, int numNodes,
                                     const KeyType* leaves, const TreeNodeIndex* octToBinary,
                                     KeyType* prefixes, TreeNodeIndex* nodeOrder, TreeNodeIndex* inverseNodeOrder)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
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

template<class KeyType>
__global__ void linkTree(const BinaryNode<KeyType>* binaryTree, TreeNodeIndex numInternalNodes, const TreeNodeIndex* octToBinary,
                         const TreeNodeIndex* binaryToOct, const TreeNodeIndex* inverseNodeOrder,
                         TreeNodeIndex* childOffsets, TreeNodeIndex* parents)
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

        childOffsets[idxB]        = childB;
        parents[(childB - 1) / 8] = idxB;
    }
}

template<class KeyType>
struct compareLevelThenPrefix
{
    __thrust_exec_check_disable__ HOST_DEVICE_FUN bool operator()(KeyType a, KeyType b) const
    {
        unsigned prefix_a = decodePrefixLength(a);
        unsigned prefix_b = decodePrefixLength(b);

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
            return decodePlaceholderBit(a) < decodePlaceholderBit(b);
        }
    }
};

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

    thrust::sort_by_key(thrust::device, d.prefixes, d.prefixes + numNodes, d.nodeOrder,
                        compareLevelThenPrefix<KeyType>{});
    // arrays now in sorted layout B

    // temporarily repurpose childOffsets as space for sort key
    thrust::copy(thrust::device, d.nodeOrder, d.nodeOrder + numNodes, d.childOffsets);
    thrust::sort_by_key(thrust::device, d.childOffsets, d.childOffsets + numNodes, d.inverseNodeOrder);
    thrust::fill(thrust::device, d.childOffsets, d.childOffsets + numNodes, 0);

    linkTree<<<iceil(d.numInternalNodes, numThreads), numThreads>>>(
        d.binaryTree, d.numInternalNodes, d.octToBinary, d.binaryToOct, d.inverseNodeOrder, d.childOffsets, d.parents);
}

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
        levels.resize(numNodes);

        binaryToOct.resize(numLeafNodes);
        octToBinary.resize(numInternalNodes);

        childOffsets.resize(numNodes);
        nodeOrder.resize(numNodes);
        inverseNodeOrder.resize(numNodes);

        parents.resize((numNodes - 1) / 8);
    }

    OctreeGpuDataView<KeyType> getData()
    {
        return {numLeafNodes,
                numInternalNodes,
                thrust::raw_pointer_cast(binaryTree.data()),
                thrust::raw_pointer_cast(prefixes.data()),
                thrust::raw_pointer_cast(levels.data()),
                thrust::raw_pointer_cast(binaryToOct.data()),
                thrust::raw_pointer_cast(octToBinary.data()),
                thrust::raw_pointer_cast(nodeOrder.data()),
                thrust::raw_pointer_cast(inverseNodeOrder.data()),
                thrust::raw_pointer_cast(childOffsets.data()),
                thrust::raw_pointer_cast(parents.data()),
        };
    }

    TreeNodeIndex numLeafNodes{0};
    TreeNodeIndex numInternalNodes{0};

    thrust::device_vector<BinaryNode<KeyType>> binaryTree;

    thrust::device_vector<KeyType>  prefixes;
    thrust::device_vector<unsigned> levels;
    thrust::device_vector<TreeNodeIndex> binaryToOct;
    thrust::device_vector<TreeNodeIndex> octToBinary;
    thrust::device_vector<TreeNodeIndex> nodeOrder;
    thrust::device_vector<TreeNodeIndex> inverseNodeOrder;
    thrust::device_vector<TreeNodeIndex> childOffsets;
    thrust::device_vector<TreeNodeIndex> parents;
};

} // namespace cstone