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

/*! \file
 * \brief Neighbor search tests
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <iostream>
#include <vector>

#include "gtest/gtest.h"

#include "cstone/findneighbors.hpp"

#include "coord_samples/random.hpp"

//! \brief simple N^2 all-to-all neighbor search
template<class T>
void all2allNeighbors(const T* x, const T* y, const T* z, const T* h, int n,
                      int *neighbors, int *neighborsCount, int ngmax)
{
    for (int i = 0; i < n; ++i)
    {
        T radius = 2 * h[i];
        T r2 = radius * radius;

        T xi = x[i], yi = y[i], zi = z[i];

        int ngcount = 0;
        for (int j = 0; j < n; ++j)
        {
            if (j == i) { continue; }
            if (ngcount < ngmax && distancesq(xi, yi, zi, x[j], y[j], z[j]) < r2)
            {
                neighbors[i * ngmax + ngcount++] = j;
            }
        }
        neighborsCount[i] = ngcount;
    }
}

void sortNeighbors(int *neighbors, int *neighborsCount, int n, int ngmax)
{
    for (int i = 0; i < n; ++i)
    {
        std::sort(neighbors + i*ngmax, neighbors + i*ngmax + neighborsCount[i]);
    }
}

TEST(FindNeighbors, treeLevel)
{
    EXPECT_EQ(3, radiusToTreeLevel(0.124, 1.));
    EXPECT_EQ(2, radiusToTreeLevel(0.126, 1.));
}


template<class T, class I>
class NeighborCheck
{
public:
    NeighborCheck(T r, int np, Box<T> b) : radius(r), n(np), box(b) {}

    void check()
    {
        using real = T;
        using CodeType = I;

        int ngmax = 100;

        real minRange = box.minExtent();
        RandomCoordinates<real, CodeType> coords(n, box);
        std::vector<T> h(n, radius/2);

        std::vector<int> neighborsRef(n * ngmax), neighborsCountRef(n);
        all2allNeighbors(coords.x().data(), coords.y().data(), coords.z().data(), h.data(), n,
                         neighborsRef.data(), neighborsCountRef.data(), ngmax);
        sortNeighbors(neighborsRef.data(), neighborsCountRef.data(), n, ngmax);

        std::vector<int> neighborsProbe(n * ngmax), neighborsCountProbe(n);
        for (int i = 0; i < n; ++i)
        {
            findNeighbors(i, coords.x().data(), coords.y().data(), coords.z().data(),
                          h.data(), box, coords.mortonCodes().data(),
                          neighborsProbe.data(), neighborsCountProbe.data(), n, ngmax);
        }
        sortNeighbors(neighborsProbe.data(), neighborsCountProbe.data(), n, ngmax);

        EXPECT_EQ(neighborsRef, neighborsProbe);
        EXPECT_EQ(neighborsCountRef, neighborsCountProbe);
    }

private:
    T   radius;
    int n;
    Box<T> box;
};

class FindNeighborsRandomUniform : public testing::TestWithParam<std::tuple<double, int, Box<double>>>
{
public:
    template<class I>
    void check()
    {
        double radius     = std::get<0>(GetParam());
        int    nParticles = std::get<1>(GetParam());
        Box<double> box = std::get<2>(GetParam());
        {
            NeighborCheck<double, I> chk(radius, nParticles, box);
            chk.check();
        }
    }
};

TEST_P(FindNeighborsRandomUniform, all2allComparison32bit)
{
    check<uint32_t>();
}

TEST_P(FindNeighborsRandomUniform, all2allComparison64bit)
{
    check<uint64_t>();
}

std::array<double, 2> radii{0.124, 0.0624};
std::array<int, 1>    nParticles{5000};
std::array<Box<double>, 2> boxes{{ {0.,1.,0.,1.,0.,1.},
                                           {-1.2, 0.23, -0.213, 3.213, -5.1, 1.23} }};

INSTANTIATE_TEST_SUITE_P(RandomUniformNeighbors,
                         FindNeighborsRandomUniform,
                         testing::Combine(testing::ValuesIn(radii),
                                          testing::ValuesIn(nParticles),
                                          testing::ValuesIn(boxes)));
