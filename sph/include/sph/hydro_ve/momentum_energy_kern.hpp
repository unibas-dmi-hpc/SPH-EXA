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
 * @brief Pressure gradients and energy kernel
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
momentumAndEnergyJLoop(int i, T sincIndex, T K, const cstone::Box<T>& box, const int* neighbors, int neighborsCount,
                       const T* x, const T* y, const T* z, const T* vx, const T* vy, const T* vz, const T* h,
                       const T* m, const T* prho, const T* c, const T* c11, const T* c12, const T* c13, const T* c22,
                       const T* c23, const T* c33, const T Atmin, const T Atmax, const T ramp, const T* wh,
                       const T* whd, const T* kx, const T* xm, const T* alpha, T* grad_P_x, T* grad_P_y, T* grad_P_z,
                       T* du, T* maxvsignal)
{
    T xi  = x[i];
    T yi  = y[i];
    T zi  = z[i];
    T vxi = vx[i];
    T vyi = vy[i];
    T vzi = vz[i];

    T hi  = h[i];
    T mi  = m[i];
    T ci  = c[i];
    T kxi = kx[i];

    T alpha_i = alpha[i];

    T xmassi = xm[i];
    T rhoi   = kxi * mi / xmassi;
    T prhoi  = prho[i];
    T voli   = xmassi / kxi;

    T mark_ramp = 0.0;
    T a_mom, b_mom, sigma_ij;

    T hiInv  = T(1) / hi;
    T hiInv3 = hiInv * hiInv * hiInv;

    T maxvsignali = 0.0;
    T momentum_x = 0.0, momentum_y = 0.0, momentum_z = 0.0, energy = 0.0;
    T a_visc_energy = 0.0;

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

        T vx_ij = vxi - vx[j];
        T vy_ij = vyi - vy[j];
        T vz_ij = vzi - vz[j];

        T hj    = h[j];
        T hjInv = T(1) / hj;

        T v1 = dist * hiInv;
        T v2 = dist * hjInv;

        T rv = rx * vx_ij + ry * vy_ij + rz * vz_ij;

        T hjInv3 = hjInv * hjInv * hjInv;
        T Wi     = hiInv3 * math::pow(lt::wharmonic_lt_with_derivative(wh, whd, v1), (int)sincIndex);
        T Wj     = hjInv3 * math::pow(lt::wharmonic_lt_with_derivative(wh, whd, v2), (int)sincIndex);

        T termA1_i = -(c11i * rx + c12i * ry + c13i * rz) * Wi;
        T termA2_i = -(c12i * rx + c22i * ry + c23i * rz) * Wi;
        T termA3_i = -(c13i * rx + c23i * ry + c33i * rz) * Wi;

        T c11j = c11[j];
        T c12j = c12[j];
        T c13j = c13[j];
        T c22j = c22[j];
        T c23j = c23[j];
        T c33j = c33[j];

        T termA1_j = -(c11j * rx + c12j * ry + c13j * rz) * Wj;
        T termA2_j = -(c12j * rx + c22j * ry + c23j * rz) * Wj;
        T termA3_j = -(c13j * rx + c23j * ry + c33j * rz) * Wj;

        T mj     = m[j];
        T cj     = c[j];
        T kxj    = kx[j];
        T xmassj = xm[j];
        T rhoj   = kxj * mj / xmassj;

        T alpha_j = alpha[j];

        T wij          = rv / dist;
        T viscosity_ij = artificial_viscosity(alpha_i, alpha_j, ci, cj, wij);
        T viscosity_jj = viscosity_ij;

        // For time-step calculations
        T vijsignal = ci + cj - T(3) * wij;
        maxvsignali = (vijsignal > maxvsignali) ? vijsignal : maxvsignali;

        T prhoj = prho[j];

        T Atwood = (std::abs(rhoi - rhoj)) / (rhoi + rhoj);
        if (Atwood < Atmin)
        {
            a_mom = mj * xmassi * xmassi;
            b_mom = mj * xmassj * xmassj;
        }
        else if (Atwood > Atmax)
        {
            a_mom = mj * xmassi * xmassj;
            b_mom = a_mom;
            mark_ramp += T(1);
        }
        else
        {
            sigma_ij = ramp * (Atwood - Atmin);
            a_mom    = mj * pow(xmassi, T(2) - sigma_ij) * pow(xmassj, sigma_ij);
            b_mom    = mj * pow(xmassj, T(2) - sigma_ij) * pow(xmassi, sigma_ij);
            mark_ramp += sigma_ij;
        }

        T volj       = xmassj / kxj;
        T a_visc     = voli * mj / mi * viscosity_ij;
        T b_visc     = volj * viscosity_jj;
        T momentum_i = prhoi * a_mom; // + 0.5 * a_visc;
        T momentum_j = prhoj * b_mom; // + 0.5 * b_visc;
        T a_visc_x   = T(0.5) * (a_visc * termA1_i + b_visc * termA1_j);
        T a_visc_y   = T(0.5) * (a_visc * termA2_i + b_visc * termA2_j);
        T a_visc_z   = T(0.5) * (a_visc * termA3_i + b_visc * termA3_j);

        momentum_x += momentum_i * termA1_i + momentum_j * termA1_j + a_visc_x;
        momentum_y += momentum_i * termA2_i + momentum_j * termA2_j + a_visc_y;
        momentum_z += momentum_i * termA3_i + momentum_j * termA3_j + a_visc_z;

        a_visc_energy += a_visc_x * vx_ij + a_visc_y * vy_ij + a_visc_z * vz_ij;
        energy += momentum_i * (vx_ij * termA1_i + vy_ij * termA2_i + vz_ij * termA3_i);
    }

    a_visc_energy = stl::max(T(0), a_visc_energy);
    du[i]         = K * (energy + T(0.5) * a_visc_energy); // factor of 2 already removed from 2P/rho

    // grad_P_xyz is stored as the acceleration, accel = -grad_P / rho
    grad_P_x[i] = -K * momentum_x;
    grad_P_y[i] = -K * momentum_y;
    grad_P_z[i] = -K * momentum_z;
    *maxvsignal = maxvsignali;
}

} // namespace sph
