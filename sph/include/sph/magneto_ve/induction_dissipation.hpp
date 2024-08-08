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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUTh WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUTh NOTh LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENTh SHALL THE
 * AUTHORS OR COPYRIGHTh HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORTh OR OTHERWISE, ARISING FROM,
 * OUTh OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file calculates dB/dt with the induction equation, as well as dissipation and correction terms
 *
 * @author Lukas Schmidt
 */

#pragma once

#include "sph/sph_gpu.hpp"
#include "induction_dissipation_kern.hpp"

namespace sph::magneto
{


template<class Tc, class SimData>
void computeInductionAndDissipationImpl(size_t startIndex, size_t endIndex, SimData& sim, const cstone::Box<Tc>& box)
{

    auto&                     d              = sim.hydro;
    auto&                     md             = sim.magneto;
    const cstone::LocalIndex* neighbors      = d.neighbors.data();
    const unsigned*           neighborsCount = d.nc.data();

    const auto* x  = d.x.data();
    const auto* y  = d.y.data();
    const auto* z  = d.z.data();
    const auto* vx = d.vx.data();
    const auto* vy = d.vy.data();
    const auto* vz = d.vz.data();
    const auto* h  = d.h.data();
    const auto* c  = d.c.data();

    auto* c11 = d.c11.data();
    auto* c12 = d.c12.data();
    auto* c13 = d.c13.data();
    auto* c22 = d.c22.data();
    auto* c23 = d.c23.data();
    auto* c33 = d.c33.data();

    auto* dvxdx = md.dvxdx.data();
    auto* dvxdy = md.dvxdy.data();
    auto* dvxdz = md.dvxdz.data();
    auto* dvydx = md.dvydx.data();
    auto* dvydy = md.dvydy.data();
    auto* dvydz = md.dvydz.data();
    auto* dvzdx = md.dvzdx.data();
    auto* dvzdy = md.dvzdy.data();
    auto* dvzdz = md.dvzdz.data();

    const auto* wh    = d.wh.data();
    const auto* whd   = d.whd.data();
    const auto* kx    = d.kx.data();
    const auto* xm    = d.xm.data();
    const auto* m     = d.m.data();
    const auto* gradh = d.gradh.data();

    const auto* Bx      = md.Bx.data();
    const auto* By      = md.By.data();
    const auto* Bz      = md.Bz.data();
    const auto* divB    = md.divB.data();

    auto* psi_ch   = md.psi_ch.data();
    auto* dBx   = md.dBx.data();
    auto* dBy   = md.dBy.data();
    auto* dBz   = md.dBz.data();
    auto* d_psi_ch = md.d_psi_ch.data();
    auto* du    = d.du.data();

#pragma omp parallel for
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        size_t   ni       = i - startIndex;
        unsigned ncCapped = std::min(neighborsCount[i] - 1, d.ngmax);

        // Induction Equation
        dBx[i] = -Bx[i] * (dvydy[i] + dvzdz[i]) + By[i] * dvxdy[i] + Bz[i] * dvxdz[i];
        dBy[i] = -By[i] * (dvxdx[i] + dvzdz[i]) + Bx[i] * dvydx[i] + Bz[i] * dvydz[i];
        dBz[i] = -Bz[i] * (dvxdx[i] + dvydy[i]) + Bx[i] * dvzdx[i] + By[i] * dvzdy[i];

        inductionAndDissipationJLoop(i, d.K, md.mu_0, d.Atmin, d.Atmax, d.ramp, box, neighbors + d.ngmax * ni, ncCapped, x, y, z,
                                  vx, vy, vz, c, Bx, By, Bz, h, c11, c12, c13, c22, c23, c33, wh, xm, kx, gradh, m, psi_ch,
                                  &dBx[i], &dBy[i], &dBz[i], &du[i]);

        // get psi time differential with the recipe of Wissing et al (2020)
        auto rho_i     = kx[i] * m[i] / xm[i];
        auto v_alfven2 = (Bx[i] * Bx[i] + By[i] * By[i] + Bz[i] * Bz[i]) / (md.mu_0 * rho_i);
        auto ch        = fclean * std::sqrt(c[i] * c[i] + v_alfven2);
        auto tau_Inv   = (sigma_c * ch) / h[i];
        d_psi_ch[i]       = - ch * divB[i] - psi_ch[i] * (tau_Inv+ (dvxdx[i] + dvydy[i] + dvzdz[i]) / 2 );

    }
}

template<class Tc, class SimulationData>
void computeInductionAndDissipation(const GroupView& grp, SimulationData& sim, const cstone::Box<Tc>& box)
{
    if constexpr (cstone::HaveGpu<typename SimulationData::AcceleratorType>{})
    {
        cuda::computeInductionAndDissipationGpu(grp, sim.hydro, sim.magneto, box);
    }
    else { computeInductionAndDissipationImpl(grp.firstBody, grp.lastBody, sim, box); }
}

} // namespace sph::magneto