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

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include "cstone/traversal/collisions_gpu.h"
#include "cstone/tree/update_gpu.cuh"
#include "cstone/tree/octree_gpu.h"

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
        while (!updateOctreeGpu(rawPtr(particleCodes), rawPtr(particleCodes) + numParticles, bucketSize, tree, counts,
                                tmpTree, workArray))
            ;
    };

    float buildTime = timeGpu(fullBuild);
    std::cout << "build time from scratch " << buildTime / 1000 << " nNodes(tree): " << nNodes(tree)
              << " count: " << thrust::reduce(counts.begin(), counts.end(), 0) << std::endl;

    auto updateTree = [&]()
    {
        updateOctreeGpu(rawPtr(particleCodes), rawPtr(particleCodes) + numParticles, bucketSize, tree, counts, tmpTree,
                        workArray);
    };

    float updateTime = timeGpu(updateTree);
    std::cout << "build time with guess " << updateTime / 1000 << " nNodes(tree): " << nNodes(tree)
              << " count: " << thrust::reduce(counts.begin(), counts.end(), 0) << std::endl;

    // internal tree benchmark

    OctreeData<KeyType, GpuTag> octree;
    octree.resize(nNodes(tree));
    auto buildInternal = [&]() { buildOctreeGpu(rawPtr(tree), octree.data()); };

    float internalBuildTime                   = timeGpu(buildInternal);
    thrust::host_vector<TreeNodeIndex> ranges = octree.levelRange;
    std::cout << "internal build time " << internalBuildTime / 1000 << std::endl;
    std::cout << "level ranges: ";
    for (int i = 0; i <= maxTreeLevel<KeyType>{}; ++i)
        std::cout << ranges[i] << " ";
    std::cout << std::endl;

    // halo discovery benchmark

    thrust::device_vector<float> haloRadii(nNodes(tree), 0.01);
    thrust::device_vector<int> flags(nNodes(tree), 0);

    auto octreeView      = octree.data();
    auto findHalosLambda = [octree = octreeView, &box, &tree, &haloRadii, &flags]()
    {
        findHalosGpu(octree.prefixes, octree.childOffsets, octree.internalToLeaf, rawPtr(tree), rawPtr(haloRadii), box,
                     0, octree.numLeafNodes / 4, rawPtr(flags));
    };

    float findTime = timeGpu(findHalosLambda);
    std::cout << "halo discovery " << findTime / 1000 << " nNodes(tree): " << nNodes(tree)
              << " count: " << thrust::reduce(flags.begin(), flags.end(), 0) << std::endl;

    {
        thrust::host_vector<KeyType> prefixes           = octree.prefixes;
        thrust::host_vector<TreeNodeIndex> childOffsets = octree.childOffsets;
        thrust::host_vector<TreeNodeIndex> toInternal   = octree.leafToInternal;
        thrust::host_vector<KeyType> h_tree             = tree;
        Octree<KeyType> h_octree;
        h_octree.update(h_tree.data(), nNodes(h_tree));

        thrust::host_vector<float> radii = haloRadii;
        std::vector<int> h_flags(nNodes(tree), 0);

        OctreeView<KeyType> o = h_octree.data();

        auto findHalosCpuLambda = [&]()
        {
            findHalos(o.prefixes, o.childOffsets, o.internalToLeaf, h_tree.data(), radii.data(), box, 0,
                      nNodes(tree) / 4, h_flags.data());
        };
        float findTimeCpu = timeCpu(findHalosCpuLambda);
        std::cout << "CPU halo discovery " << findTimeCpu << " nNodes(tree): " << nNodes(h_tree)
                  << " count: " << thrust::reduce(h_flags.begin(), h_flags.end(), 0) << std::endl;
    }
}
