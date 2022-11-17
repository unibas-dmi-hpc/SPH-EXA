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

#include "coord_samples/random.hpp"
#include "all_to_all.hpp"

using namespace cstone;

template<class Coordinates, class T>
static void neighborCheck(const Coordinates& coords, T radius, const Box<T>& box)
{
    using KeyType                   = typename Coordinates::KeyType;
    using KeyInt                    = typename KeyType::ValueType;
    cstone::LocalIndex numParticles = coords.x().size();
    unsigned ngmax                  = numParticles;
    unsigned bucketSize             = 64;

    std::vector<T> h(numParticles, radius / 2);

    std::vector<LocalIndex> neighborsRef(numParticles * ngmax);
    std::vector<unsigned> ncRef(numParticles);
    all2allNeighbors(coords.x().data(), coords.y().data(), coords.z().data(), h.data(), numParticles,
                     neighborsRef.data(), ncRef.data(), ngmax, box);
    sortNeighbors(neighborsRef.data(), ncRef.data(), numParticles, ngmax);

    auto [csTree, counts] =
        computeOctree(coords.particleKeys().data(), coords.particleKeys().data() + numParticles, bucketSize);

    Octree<KeyInt> octree;
    octree.update(csTree.data(), nNodes(csTree));

    std::vector<LocalIndex> layout(nNodes(csTree) + 1);
    std::exclusive_scan(counts.begin(), counts.end() + 1, layout.begin(), 0);

    EXPECT_EQ(layout.back(), numParticles);

    std::vector<Vec3<T>> centers(octree.numTreeNodes()), sizes(octree.numTreeNodes());
    gsl::span<const KeyInt> nodeKeys(octree.nodeKeys().data(), octree.numTreeNodes());
    nodeFpCenters<KeyType>(nodeKeys, centers.data(), sizes.data(), box);

    std::vector<LocalIndex> neighbors(numParticles * ngmax);
    std::vector<unsigned> nc(numParticles);

#pragma omp parallel for
    for (LocalIndex idx = 0; idx < numParticles; ++idx)
    {
        findNeighborsT(idx, coords.x().data(), coords.y().data(), coords.z().data(), h.data(),
                       octree.childOffsets().data(), octree.toLeafOrder().data(), layout.data(), centers.data(),
                       sizes.data(), box, ngmax, neighbors.data() + idx * ngmax, nc.data() + idx);
    }
    sortNeighbors(neighbors.data(), nc.data(), numParticles, ngmax);

    EXPECT_EQ(neighborsRef, neighbors);
    EXPECT_EQ(ncRef, nc);
}

class FindNeighborsTRandom
    : public testing::TestWithParam<std::tuple<double, int, std::array<double, 6>, cstone::BoundaryType>>
{
public:
    template<class KeyType, template<class...> class CoordinateKind>
    void check()
    {
        double radius                = std::get<0>(GetParam());
        int nParticles               = std::get<1>(GetParam());
        std::array<double, 6> limits = std::get<2>(GetParam());
        cstone::BoundaryType usePbc  = std::get<3>(GetParam());
        Box<double> box{limits[0], limits[1], limits[2], limits[3], limits[4], limits[5], usePbc, usePbc, usePbc};

        CoordinateKind<double, KeyType> coords(nParticles, box);

        neighborCheck(coords, radius, box);
    }
};

TEST_P(FindNeighborsTRandom, MortonUniform32) { check<MortonKey<uint32_t>, RandomCoordinates>(); }
// TEST_P(FindNeighborsRandom, MortonUniform64)   { check<MortonKey<uint64_t>, RandomCoordinates>(); }
// TEST_P(FindNeighborsRandom, MortonGaussian32)  { check<MortonKey<uint32_t>, RandomGaussianCoordinates>(); }
// TEST_P(FindNeighborsRandom, MortonGaussian64)  { check<MortonKey<uint64_t>, RandomGaussianCoordinates>(); }
TEST_P(FindNeighborsTRandom, HilbertUniform32) { check<HilbertKey<uint32_t>, RandomCoordinates>(); }
// TEST_P(FindNeighborsRandom, HilbertUniform64)  { check<HilbertKey<uint64_t>, RandomCoordinates>(); }
TEST_P(FindNeighborsTRandom, HilbertGaussian32) { check<HilbertKey<uint32_t>, RandomGaussianCoordinates>(); }
TEST_P(FindNeighborsTRandom, HilbertGaussian64) { check<HilbertKey<uint64_t>, RandomGaussianCoordinates>(); }

static std::array<double, 2> radii{0.124, 0.0624};
static std::array<int, 1> nParticles{2500};
static std::array<std::array<double, 6>, 2> boxes{{{0., 1., 0., 1., 0., 1.}, {-1.2, 0.23, -0.213, 3.213, -5.1, 1.23}}};
static std::array<cstone::BoundaryType, 2> pbcUsage{BoundaryType::open, BoundaryType::periodic};

INSTANTIATE_TEST_SUITE_P(RandomNeighbors,
                         FindNeighborsTRandom,
                         testing::Combine(testing::ValuesIn(radii),
                                          testing::ValuesIn(nParticles),
                                          testing::ValuesIn(boxes),
                                          testing::ValuesIn(pbcUsage)));

INSTANTIATE_TEST_SUITE_P(RandomNeighborsLargeRadius,
                         FindNeighborsTRandom,
                         testing::Combine(testing::Values(3.0),
                                          testing::Values(500),
                                          testing::ValuesIn(boxes),
                                          testing::ValuesIn(pbcUsage)));