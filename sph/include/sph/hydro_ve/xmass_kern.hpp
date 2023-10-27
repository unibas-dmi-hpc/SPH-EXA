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
 * @brief Generalized volume elements
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 */

#pragma once

#include "cstone/cuda/annotation.hpp"
#include "cstone/sfc/box.hpp"

#include "sph/kernels.hpp"
#include "sph/table_lookup.hpp"

namespace sph
{

//! @brief a particular choice of defining generalized volume elements
template<class T, class Tm>
HOST_DEVICE_FUN inline T veDefinition(Tm mass, T rhoZero)
{
    return mass / rhoZero;
}

template<size_t stride = 1, class Tc, class Tm, class T>
HOST_DEVICE_FUN inline T xmassJLoop(cstone::LocalIndex i, Tc K, const cstone::Box<Tc>& box,
                                    const cstone::LocalIndex* neighbors, unsigned neighborsCount, const Tc* x,
                                    const Tc* y, const Tc* z, const T* h, const Tm* m, const T* wh, const T* /*whd*/)
{
    auto xi = x[i];
    auto yi = y[i];
    auto zi = z[i];
    auto hi = h[i];
    auto mi = m[i];

    T hInv  = 1.0 / hi;
    T h3Inv = hInv * hInv * hInv;

    // initialize with self-contribution
    T rho0i = mi;
    for (unsigned pj = 0; pj < neighborsCount; ++pj)
    {
        cstone::LocalIndex j = neighbors[stride * pj];

        T dist = distancePBC(box, hi, xi, yi, zi, x[j], y[j], z[j]);
        T vloc = dist * hInv;
        T w    = lt::lookup(wh, vloc);

        rho0i += w * m[j];
    }

    T xmassi = veDefinition(mi, rho0i * K * h3Inv);
    return xmassi;
}

} // namespace sph
