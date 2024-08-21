/*
 * MIT License
 *
 * Copyright (c) 2024 CSCS, ETH Zurich, University of Basel, University of Zurich
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
 * @brief Integral-approach-to-derivative and divergence/curl i-loop driver
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 */

#pragma once

#include "sph/sph_gpu.hpp"
#include "full_divv_curlv_kern.hpp"
#include "divB_curlB_kern.hpp"

namespace sph::magneto
{

template<class Tc, class SimulationData>
void computeIadFullDivvCurlvImpl(size_t startIndex, size_t endIndex, SimulationData& sim, const cstone::Box<Tc>& box)
{
    auto&                     d              = sim.hydro;
    auto&                     m              = sim.magneto;
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

    auto* dvxdx = m.dvxdx.data();
    auto* dvxdy = m.dvxdy.data();
    auto* dvxdz = m.dvxdz.data();
    auto* dvydx = m.dvydx.data();
    auto* dvydy = m.dvydy.data();
    auto* dvydz = m.dvydz.data();
    auto* dvzdx = m.dvzdx.data();
    auto* dvzdy = m.dvzdy.data();
    auto* dvzdz = m.dvzdz.data();

    const auto* wh    = d.wh.data();
    const auto* whd   = d.whd.data();
    const auto* kx    = d.kx.data();
    const auto* xm    = d.xm.data();
    const auto* gradh = d.gradh.data();

    const auto* Bx = m.Bx.data();
    const auto* By = m.By.data();
    const auto* Bz = m.Bz.data();

    auto* divv    = d.divv.data();
    auto* curlv   = (d.x.size() == d.curlv.size()) ? d.curlv.data() : nullptr;
    auto* divB    = m.divB.data();
    auto* curlB_x = m.curlB_x.data();
    auto* curlB_y = m.curlB_y.data();
    auto* curlB_z = m.curlB_z.data();

#pragma omp parallel for
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        size_t   ni       = i - startIndex;
        unsigned ncCapped = std::min(neighborsCount[i] - 1, d.ngmax);

        IADJLoop(i, d.K, box, neighbors + d.ngmax * ni, ncCapped, x, y, z, h, wh, whd, xm, kx, c11, c12, c13, c22, c23,
                 c33);

        full_divV_curlVJLoop(i, d.K, box, neighbors + d.ngmax * ni, ncCapped, x, y, z, vx, vy, vz, h, c11, c12, c13,
                             c22, c23, c33, wh, whd, gradh, kx, xm, divv, curlv, dvxdx, dvxdy, dvxdz, dvydx, dvydy,
                             dvydz, dvzdx, dvzdy, dvzdz);

        divB_curlB_JLoop(i, d.K, box, neighbors + d.ngmax * ni, ncCapped, x, y, z, Bx, By, Bz, h, c11, c12, c13, c22,
                         c23, c33, wh, gradh, kx, xm, divB, curlB_x, curlB_y, curlB_z);
    }
}

template<class Tc, class SimulationData>
void computeIadFullDivvCurlv(const GroupView& grp, SimulationData& sim, const cstone::Box<Tc>& box)
{
    if constexpr (cstone::HaveGpu<typename SimulationData::AcceleratorType>{})
    {
        cuda::computeIadFullDivvCurlv(grp, sim.hydro, sim.magneto, box);
    }
    else { computeIadFullDivvCurlvImpl(grp.firstBody, grp.lastBody, sim, box); }
}

} // namespace sph::magneto
