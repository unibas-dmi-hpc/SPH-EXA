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
 * @brief internal octree GPU testing
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 */

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "gtest/gtest.h"

#include "cstone/tree/octree_internal.cuh"
#include "cstone/tree/octree_util.hpp"

using namespace cstone;

/*! @brief larger test case for nodeDepth to detect multithreading issues
 *
 * Depends on binary/octree generation, so not strictly a unit test
 */
template<class KeyType>
void nodeDepthThreading()
{
    // uniform level-7 tree with 2097152 nodes
    std::vector<KeyType> leaves = makeUniformNLevelTree<KeyType>(128*128*128, 1);

    std::vector<BinaryNode<KeyType>> binaryTree(nNodes(leaves));
    createBinaryTree(leaves.data(), nNodes(leaves), binaryTree.data());

    std::vector<OctreeNode<KeyType>> octree((nNodes(leaves)-1)/7);
    std::vector<TreeNodeIndex> leafParents(nNodes(leaves));

    createInternalOctreeCpu(binaryTree.data(), nNodes(leaves), octree.data(), leafParents.data());

    // upload octree to device
    thrust::device_vector<OctreeNode<KeyType>> d_octree = octree;

    thrust::device_vector<TreeNodeIndex> d_depths(octree.size(), 0);

    constexpr int nThreads = 256;
    nodeDepthKernel<<<iceil(octree.size(), nThreads), nThreads>>>(thrust::raw_pointer_cast(d_octree.data()), octree.size(),
                                                                  thrust::raw_pointer_cast(d_depths.data()));

    // download depths from device
    thrust::host_vector<TreeNodeIndex> h_depths = d_depths;

    int maxTreeLevel = log8ceil(nNodes(leaves));
    std::vector<int> depths_reference(octree.size());
    for (TreeNodeIndex i = 0; i < octree.size(); ++i)
    {
        // in a uniform tree, level + depth == maxTreeLevel is constant for all nodes
        depths_reference[i] = maxTreeLevel - octree[i].level;
    }

    EXPECT_EQ(h_depths, depths_reference);
}

TEST(InternalOctreeGpu, nodeDepthsThreading)
{
    nodeDepthThreading<unsigned>();
    nodeDepthThreading<uint64_t>();
}

template<class KeyType>
void constructInternalOctree()
{
    // uniform level-7 tree with 2097152 nodes
    std::vector<KeyType> leaves = makeUniformNLevelTree<KeyType>(128 * 128 * 128, 1);

    // upload cornerstone tree to device
    thrust::device_vector<KeyType> d_leaves = leaves;

    OctreeGpuDataAnchor<KeyType> treeDataGpu;
    treeDataGpu.resize(nNodes(leaves));

    buildInternalOctreeGpu(thrust::raw_pointer_cast(d_leaves.data()), treeDataGpu.getData());

    thrust::host_vector<TreeNodeIndex> h_children = treeDataGpu.childOffsets;
    std::cout << h_children[0] << std::endl;
    std::cout << h_children[1] << std::endl;
    std::cout << h_children[2] << std::endl;
    std::cout << h_children[3] << std::endl;
    std::cout << h_children[4] << std::endl;
    std::cout << h_children[5] << std::endl;
    std::cout << h_children[6] << std::endl;
    std::cout << h_children[7] << std::endl;
    std::cout << h_children[8] << std::endl;
    EXPECT_EQ(h_children[0], 1);
}

TEST(InternalOctreeGpu, construct)
{
    constructInternalOctree<unsigned>();
    //constructInternalOctree<uint64_t>();
}
