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
 * @brief Generalized volume element definition i-loop driver
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 */

#pragma once

#include "sph/kernels.hpp"
#include "sph/traits.hpp"
#include "xmass_kern.hpp"
#include "sph/sph.cuh"

namespace sph
{
template<typename T, class Dataset>
void computeXMassImpl(size_t startIndex, size_t endIndex, size_t ngmax, Dataset& d, const cstone::Box<T>& box)
{
    const int* neighbors      = d.neighbors.data();
    const int* neighborsCount = d.neighborsCount.data();

    const T* h = d.h.data();
    const T* m = d.m.data();
    const T* x = d.x.data();
    const T* y = d.y.data();
    const T* z = d.z.data();

    const T* wh  = d.wh.data();
    const T* whd = d.whd.data();

    T* xm = d.xm.data();

    const T K         = d.K;
    const T sincIndex = d.sincIndex;

#pragma omp parallel for
    for (size_t i = startIndex; i < endIndex; i++)
    {
        size_t ni = i - startIndex;
        xm[i]     = xmassJLoop(i, sincIndex, K, box, neighbors + ngmax * ni, neighborsCount[i], x, y, z, h, m, wh, whd);
#ifndef NDEBUG
        if (std::isnan(xm[i]))
            printf("ERROR::Rho0(%zu) rho0 %f, position: (%f %f %f), h: %f\n", i, xm[i], x[i], y[i], z[i], h[i]);
#endif
    }
}

template<typename T, class Dataset>
void computeXMass(size_t startIndex, size_t endIndex, size_t ngmax, Dataset& d, const cstone::Box<T>& box)
{
    if constexpr (sphexa::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        cuda::computeXMass(startIndex, endIndex, ngmax, d, box);
    }
    else { computeXMassImpl(startIndex, endIndex, ngmax, d, box); }
}

} // namespace sph
