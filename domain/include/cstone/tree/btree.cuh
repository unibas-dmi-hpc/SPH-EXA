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

/*! @brief @file parallel binary radix tree construction CUDA kernel
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/primitives/math.hpp"
#include "btree.hpp"

namespace cstone
{

//! @brief see createBinaryTree
template<class KeyType>
__global__ void createBinaryTreeKernel(const KeyType* cstree, TreeNodeIndex numNodes, BinaryNode<KeyType>* binaryTree)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < numNodes) { constructInternalNode(cstree, numNodes + 1, binaryTree, tid); }
}

//! @brief convenience kernel wrapper
template<class KeyType>
void createBinaryTreeGpu(const KeyType* cstree, TreeNodeIndex numNodes, BinaryNode<KeyType>* binaryTree)
{
    constexpr int numThreads = 256;
    createBinaryTreeKernel<<<iceil(numNodes, numThreads), numThreads>>>(cstree, numNodes, binaryTree);
}

} // namespace cstone