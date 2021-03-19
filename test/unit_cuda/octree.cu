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
    auto codes = makeNLevelGrid<I>(4);
    thrust::host_vector<I>   h_codes = codes;
    thrust::device_vector<I> d_codes = h_codes;

    // tree with 512 nodes
    auto cstree = makeUniformNLevelTree<I>(8*8*8, 1);
    thrust::host_vector<I> h_cstree = cstree;
    thrust::device_vector<I> d_cstree = h_cstree;

    thrust::device_vector<unsigned> d_counts(nNodes(d_cstree));

    constexpr unsigned nThreads = 512;
    unsigned nBlocks = (nNodes(d_cstree) + nThreads - 1) / nThreads;
    computeNodeCountsKernel<<<nBlocks, nThreads>>>(thrust::raw_pointer_cast(d_cstree.data()),
                                                   thrust::raw_pointer_cast(d_counts.data()),
                                                   nNodes(d_cstree),
                                                   thrust::raw_pointer_cast(d_codes.data()),
                                                   thrust::raw_pointer_cast(d_codes.data() + d_codes.size()));

    thrust::host_vector<unsigned> h_counts = d_counts;
    thrust::host_vector<unsigned> refCounts(8*8*8, 8);

    EXPECT_EQ(h_counts, refCounts);
}
