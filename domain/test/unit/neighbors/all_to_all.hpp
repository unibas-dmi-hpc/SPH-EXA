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
 * @brief All-to-all neighbor search for use in tests as reference
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <algorithm>

#include "cstone/findneighbors.hpp"

namespace cstone
{

//! @brief simple N^2 all-to-all neighbor search
template<class T>
static void all2allNeighbors(const T* x,
                             const T* y,
                             const T* z,
                             const T* h,
                             LocalIndex n,
                             LocalIndex* neighbors,
                             unsigned* neighborsCount,
                             unsigned ngmax,
                             const Box<T>& box)
{
#pragma omp parallel for
    for (LocalIndex i = 0; i < n; ++i)
    {
        T radius = 2 * h[i];
        T r2     = radius * radius;

        T xi = x[i], yi = y[i], zi = z[i];

        unsigned ngcount = 0;
        for (LocalIndex j = 0; j < n; ++j)
        {
            if (j == i) { continue; }
            if (ngcount < ngmax && distanceSq<true>(xi, yi, zi, x[j], y[j], z[j], box) < r2)
            {
                neighbors[i * ngmax + ngcount++] = j;
            }
        }
        neighborsCount[i] = ngcount;
    }
}

static void sortNeighbors(LocalIndex* neighbors, unsigned* neighborsCount, LocalIndex n, unsigned ngmax)
{
    for (LocalIndex i = 0; i < n; ++i)
    {
        std::sort(neighbors + i * ngmax, neighbors + i * ngmax + neighborsCount[i]);
    }
}

} // namespace cstone
