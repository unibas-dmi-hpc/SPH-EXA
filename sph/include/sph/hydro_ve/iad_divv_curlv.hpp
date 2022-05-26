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
 * @brief Integral-approach-to-derivative and velocity divergence/curl i-loop driver
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 */

#pragma once

#include "iad_kern.hpp"
#include "divv_curlv_kern.hpp"
#include "sph/sph.cuh"
#include "sph/traits.hpp"

namespace sph
{

template<class T, class Dataset>
void computeIadDivvCurlvImpl(size_t startIndex, size_t endIndex, size_t ngmax, Dataset& d, const cstone::Box<T>& box)
{
    const int* neighbors      = d.neighbors.data();
    const int* neighborsCount = d.neighborsCount.data();

    const T* x  = d.x.data();
    const T* y  = d.y.data();
    const T* z  = d.z.data();
    const T* vx = d.vx.data();
    const T* vy = d.vy.data();
    const T* vz = d.vz.data();
    const T* h  = d.h.data();

    T* c11 = d.c11.data();
    T* c12 = d.c12.data();
    T* c13 = d.c13.data();
    T* c22 = d.c22.data();
    T* c23 = d.c23.data();
    T* c33 = d.c33.data();

    T* divv  = d.divv.data();
    T* curlv = d.curlv.data();

    const T* wh  = d.wh.data();
    const T* whd = d.whd.data();
    const T* kx  = d.kx.data();
    const T* xm  = d.xm.data();

    const T K         = d.K;
    const T sincIndex = d.sincIndex;

#pragma omp parallel for
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        size_t ni = i - startIndex;

        IADJLoop(i,
                 sincIndex,
                 K,
                 box,
                 neighbors + ngmax * ni,
                 neighborsCount[i],
                 x,
                 y,
                 z,
                 h,
                 wh,
                 whd,
                 xm,
                 kx,
                 c11,
                 c12,
                 c13,
                 c22,
                 c23,
                 c33);

        divV_curlVJLoop(i,
                        sincIndex,
                        K,
                        box,
                        neighbors + ngmax * ni,
                        neighborsCount[i],
                        x,
                        y,
                        z,
                        vx,
                        vy,
                        vz,
                        h,
                        c11,
                        c12,
                        c13,
                        c22,
                        c23,
                        c33,
                        wh,
                        whd,
                        kx,
                        xm,
                        divv,
                        curlv);
    }
}

template<class T, class Dataset>
void computeIadDivvCurlv(size_t startIndex, size_t endIndex, size_t ngmax, Dataset& d, const cstone::Box<T>& box)
{
    if constexpr (sphexa::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        cuda::computeIadDivvCurlv(startIndex, endIndex, ngmax, d, box);
    }
    else { computeIadDivvCurlvImpl(startIndex, endIndex, ngmax, d, box); }
}

} // namespace sph
