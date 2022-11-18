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

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/tree/definitions.h"
#include "cstone/util/reallocate.hpp"

namespace cstone
{

//! Octree GPU data view for use in kernel code
template<class KeyType>
struct OctreeView
{
    TreeNodeIndex numLeafNodes;
    TreeNodeIndex numInternalNodes;

    KeyType* prefixes;
    TreeNodeIndex* childOffsets;
    TreeNodeIndex* parents;
    TreeNodeIndex* levelRange;
    TreeNodeIndex* internalToLeaf;
    TreeNodeIndex* leafToInternal;
};

/*! @brief construct the internal octree part of a given octree leaf cell array on the GPU
 *
 * @tparam       KeyType     unsigned 32- or 64-bit integer
 * @param[in]    cstoneTree  GPU buffer with the SFC leaf cell keys
 * @param[inout] d           input:  pointers to pre-allocated GPU buffers for octree cells
 *                           output: fully linked octree
 *
 * This does not allocate memory on the GPU, (except thrust temp buffers for scans and sorting)
 */
template<class KeyType>
extern void buildInternalOctreeGpu(const KeyType* cstoneTree, OctreeView<KeyType> d);

//! @brief provides a place to live for GPU resident octree data
template<class KeyType>
class OctreeGpuData
{
public:
    void resize(TreeNodeIndex numCsLeafNodes)
    {
        numLeafNodes           = numCsLeafNodes;
        numInternalNodes       = (numLeafNodes - 1) / 7;
        TreeNodeIndex numNodes = numLeafNodes + numInternalNodes;

        lowMemReallocate(numNodes, 1.01, {}, std::tie(prefixes, internalToLeaf, leafToInternal, childOffsets));
        // +1 to accommodate nodeOffsets in FocusedOctreeCore::update when numNodes == 1
        reallocate(childOffsets, numNodes + 1, 1.01);

        reallocateDestructive(parents, (numNodes - 1) / 8, 1.01);

        //+1 due to level 0 and +1 due to the upper bound for the last level
        reallocateDestructive(levelRange, maxTreeLevel<KeyType>{} + 2, 1.01);
    }

    OctreeView<KeyType> getData()
    {
        return {numLeafNodes,    numInternalNodes,   rawPtr(prefixes),       rawPtr(childOffsets),
                rawPtr(parents), rawPtr(levelRange), rawPtr(internalToLeaf), rawPtr(leafToInternal)};
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
    thrust::device_vector<TreeNodeIndex> internalToLeaf;
    //! @brief maps leaf (cstone) order to internal level-sorted order
    thrust::device_vector<TreeNodeIndex> leafToInternal;
};

} // namespace cstone
