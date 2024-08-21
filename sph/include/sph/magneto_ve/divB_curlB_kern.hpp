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
/*! @file spatial derivatives of the magnetic field
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 * @author Lukas Schmidt
 */

#pragma once

#include "cstone/cuda/annotation.hpp"
#include "cstone/sfc/box.hpp"

#include "sph/kernels.hpp"
#include "sph/table_lookup.hpp"

namespace sph::magneto
{

template<size_t stride = 1, typename Tc, class T>
HOST_DEVICE_FUN inline void divB_curlB_JLoop(cstone::LocalIndex i, Tc K, const cstone::Box<Tc>& box,
                                             const cstone::LocalIndex* neighbors, unsigned neighborsCount, const Tc* x,
                                             const Tc* y, const Tc* z, const Tc* Bx, const Tc* By, const Tc* Bz,
                                             const T* h, const T* c11, const T* c12, const T* c13, const T* c22,
                                             const T* c23, const T* c33, const T* wh, const T* gradh, const T* kx,
                                             const T* xm, T* divB, T* curlB_x, T* curlB_y, T* curlB_z)
{
    auto xi  = x[i];
    auto yi  = y[i];
    auto zi  = z[i];
    auto Bxi = Bx[i];
    auto Byi = By[i];
    auto Bzi = Bz[i];

    auto hi  = h[i];
    auto kxi = kx[i];

    auto hiInv  = T(1) / hi;
    auto hiInv3 = hiInv * hiInv * hiInv;

    // the 3 components of these vectors will be the derivatives in x,y,z directions
    cstone::Vec3<T> dBxi{0., 0., 0.}, dByi{0., 0., 0.}, dBzi{0., 0., 0.};

    auto c11i = c11[i];
    auto c12i = c12[i];
    auto c13i = c13[i];
    auto c22i = c22[i];
    auto c23i = c23[i];
    auto c33i = c33[i];

    for (unsigned pj = 0; pj < neighborsCount; ++pj)
    {
        cstone::LocalIndex j = neighbors[stride * pj];

        T rx = xi - x[j];
        T ry = yi - y[j];
        T rz = zi - z[j];

        applyPBC(box, T(2) * hi, rx, ry, rz);

        T r2   = rx * rx + ry * ry + rz * rz;
        T dist = std::sqrt(r2);

        T Bx_ji = Bx[j] - Bxi;
        T By_ji = By[j] - Byi;
        T Bz_ji = Bz[j] - Bzi;

        T v1 = dist * hiInv;
        T Wi = lt::lookup(wh, v1);

        cstone::Vec3<T> termA;
        termA[0] = -(c11i * rx + c12i * ry + c13i * rz) * Wi;
        termA[1] = -(c12i * rx + c22i * ry + c23i * rz) * Wi;
        termA[2] = -(c13i * rx + c23i * ry + c33i * rz) * Wi;

        T xmassj = xm[j];

        dBxi += (Bx_ji * xmassj) * termA;
        dByi += (By_ji * xmassj) * termA;
        dBzi += (Bz_ji * xmassj) * termA;
    }

    T norm_kxi = K * hiInv3 / (kxi * gradh[i]);
    divB[i]    = norm_kxi * (dBxi[0] + dByi[1] + dBzi[2]);

    curlB_x[i] = norm_kxi * (dBzi[1] - dByi[2]);
    curlB_y[i] = norm_kxi * (dBxi[2] - dBzi[0]);
    curlB_z[i] = norm_kxi * (dByi[0] - dBxi[1]);
}
} // namespace sph::magneto