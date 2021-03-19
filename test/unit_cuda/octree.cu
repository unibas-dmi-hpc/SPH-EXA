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
 * @brief Cornerstone octree GPU testing
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 */

#include "gtest/gtest.h"

#include <thrust/host_vector.h>

#include "cstone/tree/octree.cuh"
#include "cstone/tree/octree_util.hpp"

using namespace cstone;

TEST(OctreeGpu, computeNodeCounts)
{
    using I = unsigned;

    // 4096 codes
    thrust::host_vector<I> h_codes   = makeNLevelGrid<I>(4);
    thrust::device_vector<I> d_codes = h_codes;

    // regular level-3 cornerstone tree with 512 leaves
    thrust::host_vector<I> h_cstree   = makeUniformNLevelTree<I>(8*8*8, 1);
    // subdivide the first level-3 node
    for (int octant = 1; octant < 8; ++octant)
        h_cstree.push_back(octant*nodeRange<I>(4));

    std::sort(begin(h_cstree), end(h_cstree));

    // create + upload tree to the device
    thrust::device_vector<I> d_cstree = h_cstree;

    thrust::device_vector<unsigned> d_counts(nNodes(d_cstree));

    constexpr unsigned nThreads = 512;
    computeNodeCountsKernel<<<iceil(nNodes(d_cstree), nThreads), nThreads>>>
    (
        thrust::raw_pointer_cast(d_cstree.data()),
        thrust::raw_pointer_cast(d_counts.data()),
        nNodes(d_cstree),
        thrust::raw_pointer_cast(d_codes.data()),
        thrust::raw_pointer_cast(d_codes.data() + d_codes.size())
    );

    // download counts from device
    thrust::host_vector<unsigned> h_counts = d_counts;

    thrust::host_vector<unsigned> refCounts(nNodes(d_cstree), 8);
    // the first 8 nodes are level-4, node count is 1, the other nodes are level-3 with node counts of 8
    for (int nodeIdx = 0; nodeIdx < 8; ++nodeIdx)
        refCounts[nodeIdx] = 1;

    EXPECT_EQ(h_counts, refCounts);
}

TEST(OctreeGpu, rebalanceDecision)
{
    using I = unsigned;

    // regular level-3 cornerstone tree with 512 leaves
    thrust::host_vector<I> h_cstree   = makeUniformNLevelTree<I>(8*8*8, 1);
    // create + upload tree to the device
    thrust::device_vector<I> d_cstree = h_cstree;

    thrust::device_vector<unsigned> d_counts(8*8*8, 1);
    // set first 8 nodes to empty
    for (int i = 0; i < 8; ++i)
        d_counts[i] = 0;

    d_counts[9] = 2;

    unsigned bucketSize = 1;
    thrust::device_vector<int> convergenceFlag(1, 0);

    thrust::device_vector<TreeNodeIndex> d_nodeOps(d_counts.size());
    constexpr unsigned nThreads = 512;
    rebalanceDecisionKernel<<<iceil(d_counts.size(), nThreads), nThreads>>>
    (
            thrust::raw_pointer_cast(d_cstree.data()),
            thrust::raw_pointer_cast(d_counts.data()),
            nNodes(d_cstree),
            bucketSize,
            thrust::raw_pointer_cast(d_nodeOps.data()),
            thrust::raw_pointer_cast(convergenceFlag.data())
    );

    // download result from device
    thrust::host_vector<TreeNodeIndex> h_nodeOps = d_nodeOps;

    thrust::host_vector<TreeNodeIndex> reference(d_counts.size(), 1);
    for (int i = 1; i < 8; ++i)
        reference[i] = 0; // merge
    reference[9] = 8; // fuse

    EXPECT_EQ(h_nodeOps, reference);
    EXPECT_NE(0, convergenceFlag[0]);
}