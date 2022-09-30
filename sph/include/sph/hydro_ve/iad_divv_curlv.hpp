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

#include "sph/sph_gpu.hpp"
#include "divv_curlv_kern.hpp"
#include "iad_kern.hpp"

namespace sph
{

template<class Tc, class Dataset>
void computeIadDivvCurlvImpl(size_t startIndex, size_t endIndex, unsigned ngmax, Dataset& d, const cstone::Box<Tc>& box)
{
    const cstone::LocalIndex* neighbors      = d.neighbors.data();
    const unsigned*           neighborsCount = d.nc.data();

    const auto* x  = d.x.data();
    const auto* y  = d.y.data();
    const auto* z  = d.z.data();
    const auto* vx = d.vx.data();
    const auto* vy = d.vy.data();
    const auto* vz = d.vz.data();
    const auto* h  = d.h.data();

    auto* c11 = d.c11.data();
    auto* c12 = d.c12.data();
    auto* c13 = d.c13.data();
    auto* c22 = d.c22.data();
    auto* c23 = d.c23.data();
    auto* c33 = d.c33.data();

    auto* divv  = d.divv.data();
    auto* curlv = d.curlv.data();

    const auto* wh  = d.wh.data();
    const auto* whd = d.whd.data();
    const auto* kx  = d.kx.data();
    const auto* xm  = d.xm.data();

    const auto K         = d.K;
    const auto sincIndex = d.sincIndex;

#pragma omp parallel for
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        size_t   ni = i - startIndex;
        unsigned nc = std::min(neighborsCount[i], ngmax);

        IADJLoop(i, sincIndex, K, box, neighbors + ngmax * ni, nc, x, y, z, h, wh, whd, xm, kx, c11, c12, c13, c22, c23,
                 c33);

        divV_curlVJLoop(i, sincIndex, K, box, neighbors + ngmax * ni, nc, x, y, z, vx, vy, vz, h, c11, c12, c13, c22,
                        c23, c33, wh, whd, kx, xm, divv, curlv);
    }
}

template<class Tc, class Dataset>
void computeIadDivvCurlv(size_t startIndex, size_t endIndex, unsigned ngmax, Dataset& d, const cstone::Box<Tc>& box)
{
    if constexpr (cstone::HaveGpu<typename Dataset::AcceleratorType>{})
    {
        cuda::computeIadDivvCurlv(startIndex, endIndex, ngmax, d, box);
    }
    else { computeIadDivvCurlvImpl(startIndex, endIndex, ngmax, d, box); }
}

} // namespace sph
