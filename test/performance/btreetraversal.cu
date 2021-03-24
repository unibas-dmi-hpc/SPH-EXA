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
 * @brief Benchmark cornerstone octree generation on the GPU
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <chrono>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include "cstone/halos/btreetraversal.hpp"
#include "cstone/tree/btree.cuh"
#include "cstone/tree/octree.cuh"
#include "cstone/tree/octree_util.hpp"

#include "coord_samples/random.hpp"

using namespace cstone;

template <class I>
CUDA_HOST_DEVICE_FUN
int countCollisions(const BinaryNode<I>* internalRoot, const I* leafNodes,
                    const IBox& collisionBox, pair<I> excludeRange)
{
    using Node    = BinaryNode<I>;
    using NodePtr = const Node*;

    NodePtr  stack[64];
    NodePtr* stackPtr = stack;

    *stackPtr++ = nullptr;

    const BinaryNode<I>* node = internalRoot;

    int collisionCount = 0;

    do
    {
        bool traverseL = traverseNode(node->child[Node::left], collisionBox, excludeRange);
        bool traverseR = traverseNode(node->child[Node::right], collisionBox, excludeRange);

        bool overlapLeafL = leafOverlap(node->leafIndex[Node::left], leafNodes, collisionBox, excludeRange);
        bool overlapLeafR = leafOverlap(node->leafIndex[Node::right], leafNodes, collisionBox, excludeRange);

        if (overlapLeafL) collisionCount++;
        if (overlapLeafR) collisionCount++;

        if (!traverseL and !traverseR)
        {
            node = *--stackPtr; // pop
        }
        else
        {
            if (traverseL && traverseR)
            {
                #ifndef __CUDA_ARCH__
                if (stackPtr-stack >= 64)
                {
                    throw std::runtime_error("btree traversal stack exhausted\n");
                }
                #endif
                *stackPtr++ = node->child[Node::right]; // push
            }

            node = (traverseL) ? node->child[Node::left] : node->child[Node::right];
        }

    } while (node != nullptr);

    return collisionCount;
}


template <class I>
__global__ void countCollisionsKernel(const BinaryNode<I>* internalRoot, const I* leafNodes,
                                      int sideLength, unsigned* nCollisionsPerLeaf)
{
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tidz = blockDim.z * blockIdx.z + threadIdx.z;
    int tid = sideLength * sideLength * tidx + sideLength * sideLength * tidy + tidz;

    int cubeLength = 1024 / sideLength;

    int ixmin = tidx * cubeLength - 1;
    int ixmax = tidx * cubeLength + cubeLength + 1;
    int iymin = tidy * cubeLength - 1;
    int iymax = tidy * cubeLength + cubeLength + 1;
    int izmin = tidz * cubeLength - 1;
    int izmax = tidz * cubeLength + cubeLength + 1;

    IBox ibox(ixmin, ixmax, iymin, iymax, izmin, izmax);
    nCollisionsPerLeaf[tid] = countCollisions(internalRoot, leafNodes, ibox, {0, 0});
}

template <class I>
void countCollisionsGpu(const BinaryNode<I>* internalRoot, const I* leafNodes,
                        TreeNodeIndex nLeaves, unsigned* nCollisionsPerLeaf)
{
    int sideLength = log8ceil(unsigned(nLeaves));

    dim3 threads(8,8,8);
    dim3 blocks(sideLength/8, sideLength/8, sideLength/8);
    countCollisionsKernel<<<blocks, threads>>>(internalRoot, leafNodes, sideLength, nCollisionsPerLeaf);
}


int main()
{
    using CodeType = unsigned;
    //Box<double> box{-1, 1};

    //int nParticles = 8000000;
    //int bucketSize = 10;

    //RandomGaussianCoordinates<double, CodeType> randomBox(nParticles, box);

    //thrust::device_vector<CodeType> tree;
    //thrust::device_vector<unsigned> counts;

    //thrust::device_vector<CodeType> particleCodes(randomBox.mortonCodes().begin(),
    //                                              randomBox.mortonCodes().end());

    //computeOctreeGpu(thrust::raw_pointer_cast(particleCodes.data()),
    //                 thrust::raw_pointer_cast(particleCodes.data() + nParticles),
    //                 bucketSize,
    //                 tree, counts);

    thrust::device_vector<CodeType> tree = makeUniformNLevelTree<CodeType>(4096, 1);

    thrust::device_vector<BinaryNode<CodeType>> binaryTree(nNodes(tree));

    auto tp0 = std::chrono::high_resolution_clock::now();
    createBinaryTreeGpu(thrust::raw_pointer_cast(tree.data()), nNodes(tree),
                        thrust::raw_pointer_cast(binaryTree.data()));
    auto tp1  = std::chrono::high_resolution_clock::now();

    double t0 = std::chrono::duration<double>(tp1 - tp0).count();
    std::cout << "binary tree build time " << t0 << std::endl;

    thrust::device_vector<unsigned> collisionCounts(nNodes(tree));

    tp0  = std::chrono::high_resolution_clock::now();
    countCollisionsGpu(thrust::raw_pointer_cast(binaryTree.data()),
                       thrust::raw_pointer_cast(tree.data()),
                       nNodes(tree),
                       thrust::raw_pointer_cast(collisionCounts.data()));
    tp1  = std::chrono::high_resolution_clock::now();
    double t1 = std::chrono::duration<double>(tp1 - tp0).count();
    std::cout << "traversal time " << t1 << " nNodes(tree): " << std::endl;

    thrust::host_vector<unsigned> h_collisionCounts = collisionCounts;
    int nfail = 0;
    for (size_t i = 0; i < h_collisionCounts.size(); ++i)
    {
        if (h_collisionCounts[i] != 27)
        {
            nfail++;
        }
    }

    std::cout << "fail count " << nfail << std::endl;
}
