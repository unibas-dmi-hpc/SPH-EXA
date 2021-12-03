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
 * @brief  Build a tree for Ryoanji with the cornerstone framework
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <vector>

#include "cstone/tree/octree_internal_td.hpp"
#include "cstone/tree/octree.cuh"
#include "cstone/tree/octree_internal.cuh"

#include "ryoanji/types.h"

template<class KeyType>
__global__ void convertTree(cstone::OctreeGpuDataView<KeyType> cstoneTree, const cstone::LocalParticleIndex* layout,
                            CellData* ryoanjiTree)
{
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < cstoneTree.numInternalNodes + cstoneTree.numLeafNodes)
    {
        cstone::LocalParticleIndex firstParticle = layout[tid];
        cstone::LocalParticleIndex lastParticle  = layout[tid + 1];

        cstone::TreeNodeIndex child = 0;
        int numChildren             = 1;

        bool isLeaf = (cstoneTree.childOffsets[tid] == 0);
        if (!isLeaf)
        {
            child       = cstoneTree.childOffsets[tid];
            numChildren = 8;
        }

        unsigned level = cstone::decodePrefixLength(cstoneTree.prefixes[tid]) / 3;
        ryoanjiTree[tid] =
            CellData(level, cstoneTree.parents[tid], firstParticle, lastParticle - firstParticle, child, numChildren);
    }
}

template<class T>
auto buildFromCstone(std::vector<util::array<T, 4>>& bodies, const Box& box, cudaVec<CellData>& ryoanjiTree)
{
    using KeyType = uint64_t;
    unsigned numParticles = bodies.size();
    unsigned bucketSize = 64;

    static_assert(std::is_same_v<float, T>);

    std::vector<T> x(numParticles);
    std::vector<T> y(numParticles);
    std::vector<T> z(numParticles);
    std::vector<T> m(numParticles);
    std::vector<KeyType> keys(numParticles);

    for (int i = 0; i < numParticles; ++i)
    {
        x[i] = bodies[i][0];
        y[i] = bodies[i][1];
        z[i] = bodies[i][2];
        m[i] = bodies[i][3];
    }

    cstone::Box<T> csBox(box.X[0] - box.R, box.X[0] + box.R,
                         box.X[1] - box.R, box.X[1] + box.R,
                         box.X[2] - box.R, box.X[2] + box.R,
                         false, false, false);

    cstone::computeSfcKeys(x.data(), y.data(), z.data(), cstone::sfcKindPointer(keys.data()), numParticles, csBox);

    std::vector<int> ordering(numParticles);
    std::iota(ordering.begin(), ordering.end(), 0);
    cstone::sort_by_key(keys.begin(), keys.end(), ordering.begin());

    cstone::reorderInPlace(ordering, bodies.data());

    thrust::device_vector<KeyType> d_particleKeys = keys;
    thrust::device_vector<KeyType> d_tree = std::vector<KeyType>{0, cstone::nodeRange<KeyType>(0)};
    thrust::device_vector<unsigned> d_counts = std::vector<unsigned>{numParticles};

    {
        thrust::device_vector<KeyType> tmpTree;
        thrust::device_vector<cstone::TreeNodeIndex> workArray;

        while (!cstone::updateOctreeGpu(thrust::raw_pointer_cast(d_particleKeys.data()),
                                        thrust::raw_pointer_cast(d_particleKeys.data()) + d_particleKeys.size(),
                                        bucketSize,
                                        d_tree,
                                        d_counts,
                                        tmpTree,
                                        workArray));
    }
    std::cout << "numNodes " << d_tree.size() << std::endl;

    thrust::host_vector<KeyType> tree = d_tree;
    thrust::host_vector<unsigned> counts = d_counts;

    std::vector<cstone::LocalParticleIndex> layout(counts.size() + 1);
    std::copy(counts.begin(), counts.end(), layout.begin());
    cstone::exclusiveScan(layout.data(), layout.size());

    cstone::TdOctree<KeyType> octree;
    octree.update(tree.data(), cstone::nNodes(tree));

    ryoanjiTree.alloc(octree.numTreeNodes(), true);

    for (int i = 0; i < octree.numTreeNodes(); ++i)
    {
        int firstParticle = 0;
        int lastParticle  = 0;
        int child = 0;
        int numChildren = 1;
        if (!octree.isLeaf(i))
        {
            child = octree.child(i, 0);
            numChildren = 8;
        }
        else
        {
            firstParticle = layout[octree.cstoneIndex(i)];
            lastParticle  = layout[octree.cstoneIndex(i) + 1];
        }
        CellData cell(
            octree.level(i), octree.parent(i), firstParticle, lastParticle - firstParticle, child, numChildren);
        ryoanjiTree[i] = cell;
    }
    ryoanjiTree.h2d();

    std::vector<int2> levelRange(cstone::maxTreeLevel<KeyType>{} + 1);
    for (int level = 0; level <= cstone::maxTreeLevel<KeyType>{}; ++level)
    {
        levelRange[level].x = octree.levelOffset(level);
        levelRange[level].y = octree.levelOffset(level + 1);
    }

    int numLevels = octree.level(octree.numTreeNodes() - 1);

    return std::make_tuple(numLevels, std::move(levelRange));
}
