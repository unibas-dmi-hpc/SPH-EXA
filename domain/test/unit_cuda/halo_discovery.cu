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
 * @brief halo discovery test on gpu
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <thrust/device_vector.h>

#include "gtest/gtest.h"

#include "cstone/halos/discovery.cuh"

#include "coord_samples/random.hpp"
#include "cstone/tree/octree_util.hpp"

using namespace cstone;

template <class KeyType>
void findHalosTest()
{
    unsigned numParticles = 100000;
    unsigned bucketSize = 16;
    Box<double> box{-1, 1};
    RandomGaussianCoordinates<double, KeyType> randomBox(numParticles, box);

    // host input data

    auto [tree, counts] =
        computeOctree(randomBox.particleKeys().data(), randomBox.particleKeys().data() + numParticles, bucketSize);

    std::vector<BinaryNode<KeyType>> binaryTree(nNodes(tree));
    createBinaryTree(tree.data(), nNodes(tree), binaryTree.data());

    std::vector<int> flags(nNodes(tree), 0);
    std::vector<float> radii(nNodes(tree), 0.01);

    // identical device input data

    thrust::device_vector<KeyType> d_tree = tree;
    thrust::device_vector<BinaryNode<KeyType>> d_binaryTree = binaryTree;
    thrust::device_vector<int> d_flags = flags;
    thrust::device_vector<float> d_radii = radii;

    TreeNodeIndex lowerNode = nNodes(tree) / 4;
    TreeNodeIndex upperNode = nNodes(tree) / 2;

    // compute halo nodes on CPU and GPU

    findHalos(tree.data(), binaryTree.data(), radii.data(), box, lowerNode, upperNode, flags.data());

    findHalosGpu(thrust::raw_pointer_cast(d_tree.data()),
                 thrust::raw_pointer_cast(d_binaryTree.data()),
                 thrust::raw_pointer_cast(d_radii.data()),
                 box, lowerNode, upperNode,
                 thrust::raw_pointer_cast(d_flags.data()));

    std::vector<int> gpuFlags(nNodes(tree));

    // download flags from device
    thrust::copy(d_flags.begin(), d_flags.end(), gpuFlags.begin());

    EXPECT_EQ(gpuFlags, flags);
}

TEST(BinaryTreeGpu, findHalos)
{
    findHalosTest<unsigned>();
    findHalosTest<uint64_t>();
}

