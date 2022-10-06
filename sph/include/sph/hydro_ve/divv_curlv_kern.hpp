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
 * @brief Divergence of velocity vector field
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 */

#pragma once

#include "cstone/cuda/annotation.hpp"
#include "cstone/sfc/box.hpp"

#include "sph/kernels.hpp"
#include "sph/math.hpp"
#include "sph/tables.hpp"

namespace sph
{

template<typename Tc, class T>
HOST_DEVICE_FUN inline void
divV_curlVJLoop(cstone::LocalIndex i, T sincIndex, T K, const cstone::Box<Tc>& box, const cstone::LocalIndex* neighbors,
                unsigned neighborsCount, const Tc* x, const Tc* y, const Tc* z, const T* vx, const T* vy, const T* vz,
                const T* h, const T* c11, const T* c12, const T* c13, const T* c22, const T* c23, const T* c33,
                const T* wh, const T* whd, const T* kx, const T* xm, T* divv, T* curlv)
{
    auto xi  = x[i];
    auto yi  = y[i];
    auto zi  = z[i];
    auto vxi = vx[i];
    auto vyi = vy[i];
    auto vzi = vz[i];
    auto hi  = h[i];

    auto hiInv  = T(1) / hi;
    auto hiInv3 = hiInv * hiInv * hiInv;

    T divvi   = 0.0;
    T curlv_x = 0.0;
    T curlv_y = 0.0;
    T curlv_z = 0.0;

    auto c11i = c11[i];
    auto c12i = c12[i];
    auto c13i = c13[i];
    auto c22i = c22[i];
    auto c23i = c23[i];
    auto c33i = c33[i];

    for (unsigned pj = 0; pj < neighborsCount; ++pj)
    {
        cstone::LocalIndex j = neighbors[pj];

        T rx = xi - x[j];
        T ry = yi - y[j];
        T rz = zi - z[j];

        applyPBC(box, T(2) * hi, rx, ry, rz);

        T r2   = rx * rx + ry * ry + rz * rz;
        T dist = std::sqrt(r2);

        T vx_ji = vx[j] - vxi;
        T vy_ji = vy[j] - vyi;
        T vz_ji = vz[j] - vzi;

        T v1 = dist * hiInv;
        T Wi = math::pow(lt::wharmonic_lt_with_derivative(wh, whd, v1), (int)sincIndex);

        T termA1 = -(c11i * rx + c12i * ry + c13i * rz) * Wi;
        T termA2 = -(c12i * rx + c22i * ry + c23i * rz) * Wi;
        T termA3 = -(c13i * rx + c23i * ry + c33i * rz) * Wi;

        T xmassj = xm[j];

        divvi += (vx_ji * termA1 + vy_ji * termA2 + vz_ji * termA3) * xmassj;

        curlv_x += (vz_ji * termA2 - vy_ji * termA3) * xmassj;
        curlv_y += (vx_ji * termA3 - vz_ji * termA1) * xmassj;
        curlv_z += (vy_ji * termA1 - vx_ji * termA2) * xmassj;
    }

    divv[i]  = K * hiInv3 * divvi / kx[i];
    curlv[i] = K * hiInv3 * std::abs(std::sqrt(curlv_x * curlv_x + curlv_y * curlv_y + curlv_z * curlv_z)) / kx[i];
}

} // namespace sph
