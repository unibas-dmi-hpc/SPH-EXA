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
 * @brief Neighbor search tests using tree traversal
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#include <span>

#include "gtest/gtest.h"
#include "cstone/findneighbors.hpp"
#include "cstone/tree/octree_internal.hpp"
#include "cstone/traversal/traversal.hpp"

#include "coord_samples/random.hpp"
#include "all_to_all.hpp"

using namespace cstone;

template<class KeyType, class T>
void nodeFpCenters(std::span<const KeyType> prefixes, Vec3<T>* centers, Vec3<T>* sizes, const Box<T>& box)
{
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < prefixes.size(); ++i)
    {
        KeyType prefix                 = prefixes[i];
        KeyType startKey               = decodePlaceholderBit(prefix);
        unsigned level                 = decodePrefixLength(prefix) / 3;
        auto nodeBox                   = sfcIBox(HilbertKey<KeyType>(startKey), level);
        std::tie(centers[i], sizes[i]) = centerAndSize<KeyType>(nodeBox, box);
    }
}

template<class T>
void traverseNeighbors(LocalIndex i,
                       const T* x,
                       const T* y,
                       const T* z,
                       const T* h,
                       const TreeNodeIndex* childOffsets,
                       const TreeNodeIndex* toLeafOrder,
                       const LocalIndex* layout,
                       const Vec3<T>* centers,
                       const Vec3<T>* sizes,
                       const Box<T>& box,
                       unsigned ngmax,
                       LocalIndex* neighbors,
                       unsigned* nc)
{
    T xi = x[i];
    T yi = y[i];
    T zi = z[i];
    T hi = h[i];

    T radiusSq = 4.0 * hi * hi;
    Vec3<T> particle{xi, yi, zi};
    unsigned numNeighbors = 0;

    auto overlaps = [particle, radiusSq, centers, sizes, &box](TreeNodeIndex idx)
    {
        auto nodeCenter = centers[idx];
        auto nodeSize   = sizes[idx];
        return norm2(minDistance(particle, nodeCenter, nodeSize, box)) < radiusSq;
    };

    auto searchBox =
        [i, particle, radiusSq, layout, toLeafOrder, x, y, z, ngmax, neighbors, &numNeighbors, &box](TreeNodeIndex idx)
    {
        TreeNodeIndex leafIdx    = toLeafOrder[idx];
        LocalIndex firstParticle = layout[leafIdx];
        LocalIndex lastParticle  = layout[leafIdx + 1];

        for (LocalIndex j = firstParticle; j < lastParticle; ++j)
        {
            if (j == i) { continue; }
            if (distanceSqPbc(x[j], y[j], z[j], particle[0], particle[1], particle[2], box) < radiusSq)
            {
                if (numNeighbors < ngmax) { neighbors[numNeighbors] = j; }
                numNeighbors++;
            }
        }
    };

    singleTraversal(childOffsets, overlaps, searchBox);

    nc[i] = numNeighbors;
}

TEST(FindNeighbors, traversal)
{
    using T       = double;
    using KeyType = uint64_t;

    unsigned numParticles = 10000;
    unsigned bucketSize   = 64;
    unsigned ngmax        = 150;

    Box<T> box(0, 1);

    //RandomGaussianCoordinates<T, HilbertKey<KeyType>> coords(numParticles, box);
    RandomCoordinates<T, HilbertKey<KeyType>> coords(numParticles, box);

    auto [csTree, counts] =
        computeOctree(coords.particleKeys().data(), coords.particleKeys().data() + numParticles, bucketSize);

    Octree<KeyType> octree;
    octree.update(csTree.data(), nNodes(csTree));

    std::vector<LocalIndex> layout(nNodes(csTree) + 1);
    std::exclusive_scan(counts.begin(), counts.end() + 1, layout.begin(), 0);

    EXPECT_EQ(layout.back(), numParticles);

    std::vector<Vec3<T>> centers(octree.numTreeNodes()), sizes(octree.numTreeNodes());

    std::span<const KeyType> nodeKeys(octree.nodeKeys().data(), octree.numTreeNodes());
    nodeFpCenters(nodeKeys, centers.data(), sizes.data(), box);

    std::vector<T> h(numParticles, 0.05);
    std::vector<LocalIndex> neighbors(numParticles * ngmax);
    std::vector<unsigned> nc(numParticles);

#pragma omp parallel for
    for (LocalIndex idx = 0; idx < numParticles; ++idx)
    {
        traverseNeighbors(idx, coords.x().data(), coords.y().data(), coords.z().data(), h.data(),
                          octree.childOffsets().data(), octree.toLeafOrder().data(), layout.data(), centers.data(),
                          sizes.data(), box, ngmax, neighbors.data() + idx * ngmax, nc.data());
    }
    sortNeighbors(neighbors.data(), nc.data(), numParticles, ngmax);

    std::vector<unsigned> ncRef(numParticles);
    std::vector<LocalIndex> neighborsRef(numParticles * ngmax);

    all2allNeighbors(coords.x().data(), coords.y().data(), coords.z().data(), h.data(), numParticles,
                     neighborsRef.data(), ncRef.data(), ngmax, box);
    sortNeighbors(neighborsRef.data(), ncRef.data(), numParticles, ngmax);

    EXPECT_EQ(neighbors, neighborsRef);
    EXPECT_EQ(nc, ncRef);
}