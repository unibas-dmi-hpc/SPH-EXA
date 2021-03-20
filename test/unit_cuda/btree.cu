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
 * @brief binary radix tree GPU testing
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 */

#include <thrust/device_vector.h>

#include "gtest/gtest.h"

#include "cstone/tree/btree.cuh"
#include "cstone/tree/octree_util.hpp"

using namespace cstone;

//! @brief check binary node prefixes
template <class I>
void internal4x4x4PrefixTest()
{
    // a tree with 4 subdivisions along each dimension, 64 nodes
    thrust::device_vector<I> tree = makeUniformNLevelTree<I>(64, 1);

    thrust::device_vector<BinaryNode<I>> d_internalTree(nNodes(tree));
    constexpr int nThreads = 512;
    createBinaryTreeKernel<<<iceil(nNodes(tree), nThreads), nThreads>>>(thrust::raw_pointer_cast(tree.data()), nNodes(tree),
                           thrust::raw_pointer_cast(d_internalTree.data()));

    thrust::host_vector<BinaryNode<I>> internalTree = d_internalTree;

    EXPECT_EQ(internalTree[0].prefixLength, 0);
    EXPECT_EQ(internalTree[0].prefix, 0);

    EXPECT_EQ(internalTree[31].prefixLength, 1);
    EXPECT_EQ(internalTree[31].prefix, pad(I(0b0), 1));
    EXPECT_EQ(internalTree[32].prefixLength, 1);
    EXPECT_EQ(internalTree[32].prefix, pad(I(0b1), 1));

    EXPECT_EQ(internalTree[15].prefixLength, 2);
    EXPECT_EQ(internalTree[15].prefix, pad(I(0b00), 2));
    EXPECT_EQ(internalTree[16].prefixLength, 2);
    EXPECT_EQ(internalTree[16].prefix, pad(I(0b01), 2));

    EXPECT_EQ(internalTree[7].prefixLength, 3);
    EXPECT_EQ(internalTree[7].prefix, pad(I(0b000), 3));
    EXPECT_EQ(internalTree[8].prefixLength, 3);
    EXPECT_EQ(internalTree[8].prefix, pad(I(0b001), 3));

    // second (useless) root node
    EXPECT_EQ(internalTree[63].prefixLength, 0);
    EXPECT_EQ(internalTree[63].prefix, 0);
}

TEST(BinaryTreeGpu, internalTree4x4x4PrefixTest)
{
    internal4x4x4PrefixTest<unsigned>();
    internal4x4x4PrefixTest<uint64_t>();
}

