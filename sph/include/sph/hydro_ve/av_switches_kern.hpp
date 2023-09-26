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
 * @brief Artifical viscosity switches
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

template<size_t stride = 1, class Tc, class T>
HOST_DEVICE_FUN inline T
AVswitchesJLoop(cstone::LocalIndex i, Tc K, const cstone::Box<Tc>& box, const cstone::LocalIndex* neighbors,
                unsigned neighborsCount, const Tc* x, const Tc* y, const Tc* z, const T* vx, const T* vy, const T* vz,
                const T* h, const T* c, const T* c11, const T* c12, const T* c13, const T* c22, const T* c23,
                const T* c33, const T* wh, const T* /*whd*/, const T* kx, const T* xm, const T* divv, const Tc dt,
                const T alphamin, const T alphamax, const T decay_constant, T alpha_i)
{
    auto xi  = x[i];
    auto yi  = y[i];
    auto zi  = z[i];
    auto vxi = vx[i];
    auto vyi = vy[i];
    auto vzi = vz[i];

    auto hi = h[i];
    auto ci = c[i];

    auto c11i = c11[i];
    auto c12i = c12[i];
    auto c13i = c13[i];
    auto c22i = c22[i];
    auto c23i = c23[i];
    auto c33i = c33[i];

    T vijsignal_i = T(1.e-40) * ci;

    auto hiInv  = T(1) / hi;
    auto hiInv3 = hiInv * hiInv * hiInv;

    auto divv_i = divv[i];

    T graddivv_x = 0.0;
    T graddivv_y = 0.0;
    T graddivv_z = 0.0;

    for (unsigned pj = 0; pj < neighborsCount; ++pj)
    {
        cstone::LocalIndex j = neighbors[stride * pj];

        T rx = xi - x[j];
        T ry = yi - y[j];
        T rz = zi - z[j];

        applyPBC(box, T(2) * hi, rx, ry, rz);

        T r2   = rx * rx + ry * ry + rz * rz;
        T dist = std::sqrt(r2);

        T vx_ij = vxi - vx[j];
        T vy_ij = vyi - vy[j];
        T vz_ij = vzi - vz[j];

        T rv           = rx * vx_ij + ry * vy_ij + rz * vz_ij;
        T vijsignal_ij = 0.0;

        if (rv < T(0)) { vijsignal_ij = ci + c[j] - T(3) * rv / dist; }
        vijsignal_i = stl::max(vijsignal_i, vijsignal_ij);

        T v1 = dist * hiInv;
        T Wi = K * hiInv3 * lt::lookup(wh, v1);

        T termA1 = -(c11i * rx + c12i * ry + c13i * rz) * Wi;
        T termA2 = -(c12i * rx + c22i * ry + c23i * rz) * Wi;
        T termA3 = -(c13i * rx + c23i * ry + c33i * rz) * Wi;

        T volj   = xm[j] / kx[j];
        T factor = volj * (divv_i - divv[j]);

        graddivv_x += factor * termA1;
        graddivv_y += factor * termA2;
        graddivv_z += factor * termA3;
    }

    T graddivv = std::sqrt(graddivv_x * graddivv_x + graddivv_y * graddivv_y + graddivv_z * graddivv_z);

    T alphaloc = 0.0;
    if (divv_i < T(0))
    {
        T a_const = hi * hi * graddivv;
        alphaloc  = alphamax * a_const / (a_const + hi * std::abs(divv_i) + T(0.05) * ci);
    }

    if (alphaloc >= alpha_i) { alpha_i = alphaloc; }
    else
    {
        T decay    = hi / (decay_constant * vijsignal_i);
        T alphadot = 0.0;
        if (alphaloc >= alphamin) { alphadot = (alphaloc - alpha_i) / decay; }
        else { alphadot = (alphamin - alpha_i) / decay; }
        alpha_i += alphadot * dt;
    }

    return alpha_i;
}

} // namespace sph
