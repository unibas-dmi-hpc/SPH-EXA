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

#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include "cstone/halos/discovery.cuh"
#include "cstone/tree/octree.cuh"
#include "cstone/tree/octree_internal.cuh"

#include "coord_samples/random.hpp"

#include "timing.cuh"

using namespace cstone;

int main()
{
    using KeyType = uint64_t;
    Box<double> box{-1, 1};

    unsigned numParticles = 2000000;
    unsigned bucketSize   = 16;

    RandomGaussianCoordinates<double, MortonKey<KeyType>> randomBox(numParticles, box);

    thrust::device_vector<KeyType> tree    = std::vector<KeyType>{0, nodeRange<KeyType>(0)};
    thrust::device_vector<unsigned> counts = std::vector<unsigned>{numParticles};

    thrust::device_vector<KeyType> tmpTree;
    thrust::device_vector<TreeNodeIndex> workArray;

    thrust::device_vector<KeyType> particleCodes(randomBox.particleKeys().begin(), randomBox.particleKeys().end());

    // cornerstone build benchmark

    auto fullBuild = [&]()
    {
        while (!updateOctreeGpu(thrust::raw_pointer_cast(particleCodes.data()),
                                thrust::raw_pointer_cast(particleCodes.data() + numParticles), bucketSize, tree, counts,
                                tmpTree, workArray))
            ;
    };

    float buildTime = timeGpu(fullBuild);
    std::cout << "build time from scratch " << buildTime / 1000 << " nNodes(tree): " << nNodes(tree)
              << " count: " << thrust::reduce(counts.begin(), counts.end(), 0) << std::endl;

    auto updateTree = [&]()
    {
        updateOctreeGpu(thrust::raw_pointer_cast(particleCodes.data()),
                        thrust::raw_pointer_cast(particleCodes.data() + numParticles), bucketSize, tree, counts,
                        tmpTree, workArray);
    };

    float updateTime = timeGpu(updateTree);
    std::cout << "build time with guess " << updateTime / 1000 << " nNodes(tree): " << nNodes(tree)
              << " count: " << thrust::reduce(counts.begin(), counts.end(), 0) << std::endl;

    // internal tree benchmark

    OctreeGpuDataAnchor<KeyType> octree;
    octree.resize(nNodes(tree));
    auto buildInternal = [&]() { buildInternalOctreeGpu(thrust::raw_pointer_cast(tree.data()), octree.getData()); };

    float internalBuildTime                   = timeGpu(buildInternal);
    thrust::host_vector<TreeNodeIndex> ranges = octree.levelRange;
    std::cout << "internal build time " << internalBuildTime / 1000 << std::endl;
    std::cout << "level ranges: ";
    for (int i = 0; i <= maxTreeLevel<KeyType>{}; ++i)
        std::cout << ranges[i] << " ";
    std::cout << std::endl;

    // halo discovery benchmark

    thrust::device_vector<BinaryNode<KeyType>> binaryTree(nNodes(tree));
    createBinaryTreeGpu(thrust::raw_pointer_cast(tree.data()), nNodes(tree),
                        thrust::raw_pointer_cast(binaryTree.data()));

    thrust::device_vector<float> haloRadii(nNodes(tree), 0.01);
    thrust::device_vector<int> flags(nNodes(tree), 0);

    auto findHalosLambda = [&]()
    {
        findHalosGpu(thrust::raw_pointer_cast(tree.data()), thrust::raw_pointer_cast(binaryTree.data()),
                     thrust::raw_pointer_cast(haloRadii.data()), box, 0, nNodes(tree) / 4,
                     thrust::raw_pointer_cast(flags.data()));
    };

    float findTime = timeGpu(findHalosLambda);
    std::cout << "halo discovery " << findTime / 1000 << " nNodes(tree): " << nNodes(tree)
              << " count: " << thrust::reduce(flags.begin(), flags.end(), 0) << std::endl;
}
