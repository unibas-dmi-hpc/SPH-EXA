/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
 *               2022 University of Basel
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
 * @brief Density i-loop OpenMP driver
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <vector>

#include "cstone/findneighbors.hpp"

#include "sph/sph_gpu.hpp"
#include "density_kern.hpp"

namespace sph
{
template<class T, class Dataset>
void computeDensityImpl(size_t startIndex, size_t endIndex, unsigned ngmax, Dataset& d, const cstone::Box<T>& box)
{
    const cstone::LocalIndex* neighbors      = d.neighbors.data();
    const unsigned*           neighborsCount = d.nc.data();

    const auto* h = d.h.data();
    const auto* m = d.m.data();
    const auto* x = d.x.data();
    const auto* y = d.y.data();
    const auto* z = d.z.data();

    const auto* wh  = d.wh.data();
    const auto* whd = d.whd.data();

    auto* rho = d.rho.data();

    const T K         = d.K;
    const T sincIndex = d.sincIndex;

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; i++)
    {
        // int neighLoc[ngmax];
        // int count;
        // cstone::findNeighbors(
        //    pi, x, y, z, h, box, cstone::sfcKindPointer(d.codes.data()), neighLoc, &count, d.codes.size(), ngmax);

        size_t ni = i - startIndex;

        unsigned nc = std::min(neighborsCount[i], ngmax);
        rho[i]      = densityJLoop(i, sincIndex, K, box, neighbors + ngmax * ni, nc, x, y, z, h, m, wh, whd);

#ifndef NDEBUG
        if (std::isnan(rho[i]))
            printf("ERROR::Density(%zu) density %f, position: (%f %f %f), h: %f\n", i, rho[i], x[i], y[i], z[i], h[i]);
#endif
    }
}

template<class T, class Dataset>
void computeDensity(size_t startIndex, size_t endIndex, unsigned ngmax, Dataset& d, const cstone::Box<T>& box)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        cuda::computeDensity(startIndex, endIndex, ngmax, d, box);
    }
    else { computeDensityImpl(startIndex, endIndex, ngmax, d, box); }
}

} // namespace sph
