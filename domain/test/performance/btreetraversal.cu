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
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include "cstone/halos/btreetraversal.hpp"
#include "cstone/tree/btree.cuh"
#include "cstone/tree/octree_util.hpp"

using namespace cstone;

template <class KeyType>
HOST_DEVICE_FUN int countCollisions(const BinaryNode<KeyType>* root, const KeyType* leafNodes,
                    const IBox& collisionBox, pair<KeyType> excludeRange)
{
    int collisionCount = 0;
    auto counter = [&collisionCount](TreeNodeIndex i) { collisionCount++; };
    findCollisions(root, leafNodes, counter, collisionBox, excludeRange);

    return collisionCount;
}


template <class KeyType>
__global__ void countCollisionsKernel(const BinaryNode<KeyType>* internalRoot, const KeyType* leafNodes,
                                      int sideLength, unsigned* nCollisionsPerLeaf)
{
    int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tidz = blockDim.z * blockIdx.z + threadIdx.z;
    int tid = sideLength * sideLength * tidz + sideLength * tidy + tidx;

    int cubeLength = 1024 / sideLength;

    int ixmin = tidx * cubeLength - 1;
    int ixmax = tidx * cubeLength + cubeLength + 1;
    int iymin = tidy * cubeLength - 1;
    int iymax = tidy * cubeLength + cubeLength + 1;
    int izmin = tidz * cubeLength - 1;
    int izmax = tidz * cubeLength + cubeLength + 1;

    IBox ibox(ixmin, ixmax, iymin, iymax, izmin, izmax);
    unsigned cnt = countCollisions(internalRoot, leafNodes, ibox, {0, 0});
    nCollisionsPerLeaf[tid] = cnt;
    //printf("%d| %d, %d, %d: %d\n", tid, tidx, tidy, tidz, cnt);
}

template <class KeyType>
void countCollisionsGpu(const BinaryNode<KeyType>* internalRoot, const KeyType* leafNodes,
                        TreeNodeIndex nLeaves, unsigned* nCollisionsPerLeaf)
{
    int sideLength = 1u << log8ceil(unsigned(nLeaves));
    std::cout << "nNodes " << nLeaves << " sideLength " << sideLength << std::endl;

    int tbl = 8; // thread block cube length
    dim3 threads(tbl,tbl,tbl);
    dim3 blocks(sideLength/tbl, sideLength/tbl, sideLength/tbl);
    countCollisionsKernel<<<blocks, threads>>>(internalRoot, leafNodes, sideLength, nCollisionsPerLeaf);
}

template <class KeyType>
void countCollisionsCpu(const BinaryNode<KeyType>* internalRoot, const KeyType* leafNodes,
                        TreeNodeIndex nLeaves, unsigned* nCollisionsPerLeaf)
{
    int sideLength = 1u << log8ceil(unsigned(nLeaves));
    std::cout << "nNodes " << nLeaves << " sideLength " << sideLength << std::endl;

    #pragma omp parallel for schedule(static)
    for (int tidx = 0; tidx < sideLength; ++tidx)
    for (int tidy = 0; tidy < sideLength; ++tidy)
    for (int tidz = 0; tidz < sideLength; ++tidz)
    {
        int tid = sideLength * sideLength * tidx + sideLength * tidy + tidz;

        int cubeLength = 1024 / sideLength;

        int ixmin = tidx * cubeLength - 1;
        int ixmax = tidx * cubeLength + cubeLength + 1;
        int iymin = tidy * cubeLength - 1;
        int iymax = tidy * cubeLength + cubeLength + 1;
        int izmin = tidz * cubeLength - 1;
        int izmax = tidz * cubeLength + cubeLength + 1;

        IBox ibox(ixmin, ixmax, iymin, iymax, izmin, izmax);
        unsigned cnt = countCollisions(internalRoot, leafNodes, ibox, {0, 0});
        nCollisionsPerLeaf[tid] = cnt;
    }
}

template<class CodeType>
void testCpu(unsigned gridSize)
{
    std::vector<CodeType> tree = makeUniformNLevelTree<CodeType>(gridSize, 1);

    std::vector<BinaryNode<CodeType>> binaryTree(nNodes(tree));

    auto tp0 = std::chrono::high_resolution_clock::now();
    createBinaryTree(tree.data(), nNodes(tree), binaryTree.data());
    auto tp1  = std::chrono::high_resolution_clock::now();

    double t0 = std::chrono::duration<double>(tp1 - tp0).count();
    std::cout << "cpu binary tree build time " << t0 << std::endl;

    std::vector<unsigned> collisionCounts(nNodes(tree));

    tp0  = std::chrono::high_resolution_clock::now();
    countCollisionsCpu(binaryTree.data(),
                       tree.data(),
                       nNodes(tree), collisionCounts.data());
    tp1  = std::chrono::high_resolution_clock::now();
    double t1 = std::chrono::duration<double>(tp1 - tp0).count();
    std::cout << "cpu traversal time " << t1 << std::endl;

    int nfail = 0;
    for (size_t i = 0; i < collisionCounts.size(); ++i)
    {
        if (collisionCounts[i] != 27)
        {
            nfail++;
        }
    }
    std::cout << "cpu fail count " << nfail << std::endl;
}

int main()
{
    using CodeType = unsigned;

    unsigned gridSize = 128*128*128;

    testCpu<CodeType>(gridSize);

    thrust::device_vector<CodeType> tree = makeUniformNLevelTree<CodeType>(gridSize, 1);

    thrust::device_vector<BinaryNode<CodeType>> binaryTree(nNodes(tree));

    auto tp0 = std::chrono::high_resolution_clock::now();
    createBinaryTreeGpu(thrust::raw_pointer_cast(tree.data()), nNodes(tree),
                        thrust::raw_pointer_cast(binaryTree.data()));
    cudaDeviceSynchronize();
    auto tp1  = std::chrono::high_resolution_clock::now();

    double t0 = std::chrono::duration<double>(tp1 - tp0).count();
    std::cout << "binary tree build time " << t0 << std::endl;

    thrust::device_vector<unsigned> collisionCounts(nNodes(tree));

    tp0  = std::chrono::high_resolution_clock::now();
    countCollisionsGpu(thrust::raw_pointer_cast(binaryTree.data()),
                       thrust::raw_pointer_cast(tree.data()),
                       nNodes(tree),
                       thrust::raw_pointer_cast(collisionCounts.data()));
    cudaDeviceSynchronize();
    tp1  = std::chrono::high_resolution_clock::now();
    double t1 = std::chrono::duration<double>(tp1 - tp0).count();
    std::cout << "traversal time " << t1 << std::endl;

    thrust::host_vector<unsigned> h_collisionCounts = collisionCounts;
    int nfail = 0;
    for (size_t i = 0; i < h_collisionCounts.size(); ++i)
    {
        if (h_collisionCounts[i] != 27)
        {
            nfail++;
            //std::cout << h_collisionCounts[i] << std::endl;
        }
    }

    std::cout << "fail count " << nfail << std::endl;
}
