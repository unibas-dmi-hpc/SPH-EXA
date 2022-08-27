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
 * @brief Neighbor search tests
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "cstone/findneighbors.hpp"
#include "coord_samples/random.hpp"

using namespace cstone;

//! @brief simple N^2 all-to-all neighbor search
template<class T>
void all2allNeighbors(const T* x,
                      const T* y,
                      const T* z,
                      const T* h,
                      int n,
                      int* neighbors,
                      int* neighborsCount,
                      int ngmax,
                      const Box<T>& box)
{
    for (int i = 0; i < n; ++i)
    {
        T radius = 2 * h[i];
        T r2     = radius * radius;

        T xi = x[i], yi = y[i], zi = z[i];

        int ngcount = 0;
        for (int j = 0; j < n; ++j)
        {
            if (j == i) { continue; }
            if (ngcount < ngmax && distanceSqPbc(xi, yi, zi, x[j], y[j], z[j], box) < r2)
            {
                neighbors[i * ngmax + ngcount++] = j;
            }
        }
        neighborsCount[i] = ngcount;
    }
}

void sortNeighbors(int* neighbors, int* neighborsCount, int n, int ngmax)
{
    for (int i = 0; i < n; ++i)
    {
        std::sort(neighbors + i * ngmax, neighbors + i * ngmax + neighborsCount[i]);
    }
}

TEST(FindNeighbors, distanceSqPbc)
{
    {
        Box<double> box(0, 10, BoundaryType::open);
        EXPECT_DOUBLE_EQ(64.0, distanceSqPbc(1., 0., 0., 9., 0., 0., box));
        EXPECT_DOUBLE_EQ(64.0, distanceSqPbc(9., 0., 0., 1., 0., 0., box));
        EXPECT_DOUBLE_EQ(192.0, distanceSqPbc(9., 9., 9., 1., 1., 1., box));
    }
    {
        Box<double> box(0, 10, BoundaryType::periodic);
        EXPECT_DOUBLE_EQ(4.0, distanceSqPbc(1., 0., 0., 9., 0., 0., box));
        EXPECT_DOUBLE_EQ(4.0, distanceSqPbc(9., 0., 0., 1., 0., 0., box));
        EXPECT_DOUBLE_EQ(12.0, distanceSqPbc(9., 9., 9., 1., 1., 1., box));
    }
}

TEST(FindNeighbors, treeLevel)
{
    EXPECT_EQ(3, radiusToTreeLevel(0.124, 1.));
    EXPECT_EQ(2, radiusToTreeLevel(0.126, 1.));
}

/*! @brief find neighbor boxes around a particles centered in (1,1,1) box
 *
 * The particles (x,y,z) is centered in the (ix,iy,iz) = (1,1,1) node
 * with i{x,y,z} = coordinates in [0, 2^maxTreeLevel<KeyType>{}]
 * The minimum radius to hit all neighboring (ix+-1,iy+-1,iz+-1) nodes is sqrt(3/4)
 * and this is checked.
 */
template<class KeyType>
void findNeighborBoxesInterior()
{
    using T = double;
    // smallest octree cell edge length in unit cube
    constexpr T uL = T(1.) / (1u << maxTreeLevel<KeyType>{});

    Box<T> bbox(0, 1);

    T x            = 1.5 * uL;
    T y            = 1.5 * uL;
    T z            = 1.5 * uL;
    T radius       = 0.867 * uL;
    T radiusSq     = radius * radius;
    unsigned level = radiusToTreeLevel(radius, bbox.minExtent());

    KeyType neighborCodes[27];
    auto pbi   = findNeighborBoxes(x, y, z, radiusSq, level, bbox, neighborCodes);
    int nBoxes = pbi[0];

    EXPECT_EQ(nBoxes, 27);
    std::sort(neighborCodes, neighborCodes + nBoxes);

    std::vector<KeyType> refBoxes;
    for (int ix = 0; ix < 3; ++ix)
        for (int iy = 0; iy < 3; ++iy)
            for (int iz = 0; iz < 3; ++iz)
            {
                refBoxes.push_back(iSfcKey<KeyType>(ix, iy, iz));
            }
    std::sort(begin(refBoxes), end(refBoxes));

    std::vector<KeyType> probeBoxes(neighborCodes, neighborCodes + nBoxes);
    EXPECT_EQ(probeBoxes, refBoxes);

    // now, the 8 farthest corners are not hit any more
    radius   = 0.866 * uL;
    radiusSq = radius * radius;
    level    = radiusToTreeLevel(radius, bbox.minExtent());
    pbi      = findNeighborBoxes(x, y, z, radiusSq, level, bbox, neighborCodes);
    EXPECT_EQ(pbi[0], 19);
}

TEST(FindNeighbors, findNeighborBoxesInterior)
{
    findNeighborBoxesInterior<MortonKey<unsigned>>();
    findNeighborBoxesInterior<MortonKey<uint64_t>>();
    findNeighborBoxesInterior<HilbertKey<unsigned>>();
    findNeighborBoxesInterior<HilbertKey<uint64_t>>();
}

/*! @brief find neighbor boxes around a particles centered in (1,1,1) box
 *
 * The particle (x,y,z) is centered in the (ix,iy,iz) = (0,0,0) node
 * with i{x,y,z} = coordinates in [0, 2^maxTreeLevel<KeyType>{}]
 * The minimum radius to hit all neighboring (ix+1,iy+1,iz+1) nodes is sqrt(3/4)
 * and this is checked. All negative offsets correspond to non-existing boxes
 * for this case that doesn't use PBC, such that there are only 8 boxes found.
 */
template<class KeyType>
void findNeighborBoxesCorner()
{
    using T = double;
    // smallest octree cell edge length in unit cube
    constexpr T uL = T(1.) / (1u << maxTreeLevel<KeyType>{});

    Box<T> bbox(0, 1);
    T halfUnitDiagonal = 0.867; // slightly more than sqrt(3) / 2

    T x            = 0.5 * uL;
    T y            = 0.5 * uL;
    T z            = 0.5 * uL;
    T radius       = halfUnitDiagonal * uL;
    unsigned level = radiusToTreeLevel(radius, bbox.minExtent());

    KeyType neighborCodes[27];
    auto pbi   = findNeighborBoxes(x, y, z, radius * radius, level, bbox, neighborCodes);
    int nBoxes = pbi[0];

    EXPECT_EQ(nBoxes, 8);
    std::sort(neighborCodes, neighborCodes + nBoxes);

    std::vector<KeyType> refBoxes;
    for (int ix = 0; ix < 2; ++ix)
        for (int iy = 0; iy < 2; ++iy)
            for (int iz = 0; iz < 2; ++iz)
            {
                refBoxes.push_back(iSfcKey<KeyType>(ix, iy, iz));
            }
    std::sort(begin(refBoxes), end(refBoxes));

    std::vector<KeyType> probeBoxes(neighborCodes, neighborCodes + nBoxes);
    EXPECT_EQ(probeBoxes, refBoxes);
}

TEST(FindNeighbors, findNeighborBoxesCorner)
{
    findNeighborBoxesCorner<MortonKey<unsigned>>();
    findNeighborBoxesCorner<MortonKey<uint64_t>>();
    findNeighborBoxesCorner<HilbertKey<unsigned>>();
    findNeighborBoxesCorner<HilbertKey<uint64_t>>();
}

template<class KeyType>
void findNeighborBoxesUpperCorner()
{
    using T = double;
    // smallest octree cell edge length in unit cube
    constexpr T uL = T(1.) / (1u << maxTreeLevel<KeyType>{});

    Box<T> bbox(0, 1);
    T halfUnitDiagonal = 0.867; // slightly more than sqrt(3) / 2

    unsigned level    = 3;
    unsigned nodeEdge = 1u << (maxTreeLevel<KeyType>{} - level);
    T nodeEdgeF       = nodeEdge * uL;

    // point centered in the level-3 box with coordinates (0, 0, 7)
    T x      = nodeEdgeF / 2;
    T y      = nodeEdgeF / 2;
    T z      = 7.5 * nodeEdgeF;
    T radius = halfUnitDiagonal * nodeEdgeF;

    KeyType neighborCodes[27];
    auto pbi   = findNeighborBoxes(x, y, z, radius * radius, level, bbox, neighborCodes);
    int nBoxes = pbi[0];

    EXPECT_EQ(nBoxes, 8);
    std::sort(neighborCodes, neighborCodes + nBoxes);

    // all level-3 boxes that touch (0, 0, 7)
    std::vector<KeyType> refBoxes;
    for (int ix = 0; ix < 2; ++ix)
        for (int iy = 0; iy < 2; ++iy)
            for (int iz = 6; iz < 8; ++iz)
            {
                KeyType refKey = enclosingBoxCode(iSfcKey<KeyType>(ix * nodeEdge, iy * nodeEdge, iz * nodeEdge), level);
                refBoxes.push_back(refKey);
            }
    std::sort(begin(refBoxes), end(refBoxes));

    std::vector<KeyType> probeBoxes(neighborCodes, neighborCodes + nBoxes);
    EXPECT_EQ(probeBoxes, refBoxes);
}

TEST(FindNeighbors, findNeighborBoxesUpperCorner)
{
    findNeighborBoxesUpperCorner<MortonKey<unsigned>>();
    findNeighborBoxesUpperCorner<MortonKey<uint64_t>>();
    findNeighborBoxesUpperCorner<HilbertKey<unsigned>>();
    findNeighborBoxesUpperCorner<HilbertKey<uint64_t>>();
}

template<class KeyType>
void findNeighborBoxesCornerPbc()
{
    using T = double;
    // smallest octree cell edge length in unit cube
    constexpr T uL = T(1.) / (1u << maxTreeLevel<KeyType>{});

    Box<T> bbox(0, 1, BoundaryType::periodic);

    T x            = 0.5 * uL;
    T y            = 0.5 * uL;
    T z            = 0.5 * uL;
    T radius       = 0.867 * uL;
    unsigned level = radiusToTreeLevel(radius, bbox.minExtent());

    KeyType neighborCodes[27];
    auto pbi    = findNeighborBoxes(x, y, z, radius * radius, level, bbox, neighborCodes);
    int nBoxes  = pbi[0];
    int iBoxPbc = pbi[1];

    EXPECT_EQ(nBoxes, 8);
    EXPECT_EQ(iBoxPbc, 8);
}

TEST(FindNeighbors, findNeighborBoxesCornerPbc)
{
    findNeighborBoxesCornerPbc<MortonKey<unsigned>>();
    findNeighborBoxesCornerPbc<MortonKey<uint64_t>>();
    findNeighborBoxesCornerPbc<HilbertKey<unsigned>>();
    findNeighborBoxesCornerPbc<HilbertKey<uint64_t>>();
}

template<class Coordinates, class T>
void neighborCheck(const Coordinates& coords, T radius, const Box<T>& box)
{
    int n     = coords.x().size();
    int ngmax = n;

    std::vector<T> h(n, radius / 2);

    std::vector<int> neighborsRef(n * ngmax), neighborsCountRef(n);
    all2allNeighbors(coords.x().data(), coords.y().data(), coords.z().data(), h.data(), n, neighborsRef.data(),
                     neighborsCountRef.data(), ngmax, box);
    sortNeighbors(neighborsRef.data(), neighborsCountRef.data(), n, ngmax);

    std::vector<int> neighborsProbe(n * ngmax), neighborsCountProbe(n);

    auto particleKeys = (typename Coordinates::KeyType*)(coords.particleKeys().data());
    findNeighbors(coords.x().data(), coords.y().data(), coords.z().data(), h.data(), 0, n, n, box, particleKeys,
                  neighborsProbe.data(), neighborsCountProbe.data(), ngmax);
    sortNeighbors(neighborsProbe.data(), neighborsCountProbe.data(), n, ngmax);

    EXPECT_EQ(neighborsRef, neighborsProbe);
    EXPECT_EQ(neighborsCountRef, neighborsCountProbe);
}

class FindNeighborsRandom
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

TEST_P(FindNeighborsRandom, MortonUniform32) { check<MortonKey<uint32_t>, RandomCoordinates>(); }
// TEST_P(FindNeighborsRandom, MortonUniform64)   { check<MortonKey<uint64_t>, RandomCoordinates>(); }
// TEST_P(FindNeighborsRandom, MortonGaussian32)  { check<MortonKey<uint32_t>, RandomGaussianCoordinates>(); }
// TEST_P(FindNeighborsRandom, MortonGaussian64)  { check<MortonKey<uint64_t>, RandomGaussianCoordinates>(); }
TEST_P(FindNeighborsRandom, HilbertUniform32) { check<HilbertKey<uint32_t>, RandomCoordinates>(); }
// TEST_P(FindNeighborsRandom, HilbertUniform64)  { check<HilbertKey<uint64_t>, RandomCoordinates>(); }
TEST_P(FindNeighborsRandom, HilbertGaussian32) { check<HilbertKey<uint32_t>, RandomGaussianCoordinates>(); }
TEST_P(FindNeighborsRandom, HilbertGaussian64) { check<HilbertKey<uint64_t>, RandomGaussianCoordinates>(); }

std::array<double, 2> radii{0.124, 0.0624};
std::array<int, 1> nParticles{2500};
std::array<std::array<double, 6>, 2> boxes{{{0., 1., 0., 1., 0., 1.}, {-1.2, 0.23, -0.213, 3.213, -5.1, 1.23}}};
std::array<cstone::BoundaryType, 2> pbcUsage{BoundaryType::open, BoundaryType::periodic};

INSTANTIATE_TEST_SUITE_P(RandomNeighbors,
                         FindNeighborsRandom,
                         testing::Combine(testing::ValuesIn(radii),
                                          testing::ValuesIn(nParticles),
                                          testing::ValuesIn(boxes),
                                          testing::ValuesIn(pbcUsage)));

INSTANTIATE_TEST_SUITE_P(RandomNeighborsLargeRadius,
                         FindNeighborsRandom,
                         testing::Combine(testing::Values(3.0),
                                          testing::Values(500),
                                          testing::ValuesIn(boxes),
                                          testing::ValuesIn(pbcUsage)));
