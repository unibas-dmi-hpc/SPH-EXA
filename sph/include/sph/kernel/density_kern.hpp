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
 * @brief Density SPH kernel
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 */

#pragma once

#include "cstone/sfc/box.hpp"

#include "sph/tables.hpp"

namespace sphexa
{
namespace sph
{
namespace kernels
{

template<typename T>
CUDA_DEVICE_HOST_FUN inline void densityJLoop(int i, T sincIndex, T K, const cstone::Box<T>& box, const int* neighbors,
                                              int neighborsCount, const T* x, const T* y, const T* z, const T* h,
                                              const T* m, const T* wh, const T* whd, const T* rho0, const T* wrho0,
                                              T* ro, T* kx, T* whomega)
{
    T xi     = x[i];
    T yi     = y[i];
    T zi     = z[i];
    T hi     = h[i];
    T rho0i  = rho0[i];
    T wrho0i = wrho0[i];
    T xmassi = m[i] / rho0i;

    T hInv  = T(1) / hi;
    T h3Inv = hInv * hInv * hInv;

    T kxi      = 0.0;
    T whomegai = 0.0;

    for (int pj = 0; pj < neighborsCount; ++pj)
    {
        int j      = neighbors[pj];
        T   dist   = distancePBC(box, hi, xi, yi, zi, x[j], y[j], z[j]);
        T   vloc   = dist * hInv;
        T   w      = ::sphexa::math::pow(lt::wharmonic_lt_with_derivative(wh, whd, vloc), sincIndex);
        T   dw     = wharmonic_derivative(vloc, w) * sincIndex;
        T   dterh  = -(3.0 * w + vloc * dw);
        T   xmassj = m[j] / rho0[j];

        kxi += w * xmassj;
        whomegai += dterh * xmassj;
    }

    kxi      = K * (kxi + xmassi) * h3Inv;
    whomegai = K * (whomegai - 3.0 * xmassi) * h3Inv * hInv;

    T roloc  = kxi * m[i] / xmassi;
    whomegai = whomegai * m[i] / xmassi + (roloc / rho0i - K * xmassi * h3Inv) * wrho0i;

    ro[i]      = roloc;
    kx[i]      = kxi;
    whomega[i] = whomegai;
}

} // namespace kernels
} // namespace sph
} // namespace sphexa
