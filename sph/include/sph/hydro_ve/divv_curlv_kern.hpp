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

#include "cstone/sfc/box.hpp"

#include "sph/tables.hpp"

namespace sph
{

template<typename T>
CUDA_DEVICE_HOST_FUN inline void
divV_curlVJLoop(int i, T sincIndex, T K, const cstone::Box<T>& box, const int* neighbors, int neighborsCount,
                const T* x, const T* y, const T* z, const T* vx, const T* vy, const T* vz, const T* h, const T* c11,
                const T* c12, const T* c13, const T* c22, const T* c23, const T* c33, const T* wh, const T* whd,
                const T* kx, const T* xm, T* divv, T* curlv)
{
    T xi  = x[i];
    T yi  = y[i];
    T zi  = z[i];
    T vxi = vx[i];
    T vyi = vy[i];
    T vzi = vz[i];
    T hi  = h[i];

    T hiInv  = 1.0 / hi;
    T hiInv3 = hiInv * hiInv * hiInv;

    T divvi   = 0.0;
    T curlv_x = 0.0;
    T curlv_y = 0.0;
    T curlv_z = 0.0;

    T c11i = c11[i];
    T c12i = c12[i];
    T c13i = c13[i];
    T c22i = c22[i];
    T c23i = c23[i];
    T c33i = c33[i];

    for (int pj = 0; pj < neighborsCount; ++pj)
    {
        int j = neighbors[pj];

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
