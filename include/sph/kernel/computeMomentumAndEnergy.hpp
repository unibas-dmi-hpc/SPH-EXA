#pragma once

#include "cstone/sfc/box.hpp"

#include "../lookupTables.hpp"

namespace sphexa
{
namespace sph
{
namespace kernels
{

template<typename T>
CUDA_DEVICE_HOST_FUN inline void
momentumAndEnergyJLoop(int i, T sincIndex, T K, const cstone::Box<T>& box, const int* neighbors, int neighborsCount,
                       const T* x, const T* y, const T* z, const T* vx, const T* vy, const T* vz, const T* h,
                       const T* m, const T* ro, const T* p, const T* c, const T* c11, const T* c12, const T* c13,
                       const T* c22, const T* c23, const T* c33, const T* wh, const T* whd, T* grad_P_x, T* grad_P_y,
                       T* grad_P_z, T* du, T* maxvsignal)
{
    constexpr T gradh_i = 1.0;
    constexpr T gradh_j = 1.0;

    T xi  = x[i];
    T yi  = y[i];
    T zi  = z[i];
    T vxi = vx[i];
    T vyi = vy[i];
    T vzi = vz[i];

    T hi  = h[i];
    T roi = ro[i];
    T pri = p[i];
    T ci  = c[i];

    T mi_roi = m[i] / ro[i];

    T hiInv  = 1.0 / hi;
    T hiInv3 = hiInv * hiInv * hiInv;

    T maxvsignali = 0.0;
    T momentum_x = 0.0, momentum_y = 0.0, momentum_z = 0.0, energy = 0.0;

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

        applyPBC(box, 2.0 * hi, rx, ry, rz);

        T r2   = rx * rx + ry * ry + rz * rz;
        T dist = std::sqrt(r2);

        T vx_ij = vxi - vx[j];
        T vy_ij = vyi - vy[j];
        T vz_ij = vzi - vz[j];

        T hj    = h[j];
        T hjInv = 1.0 / hj;

        T v1 = dist * hiInv;
        T v2 = dist * hjInv;

        T rv = rx * vx_ij + ry * vy_ij + rz * vz_ij;

        T hjInv3 = hjInv * hjInv * hjInv;
        T Wi     = hiInv3 * ::sphexa::math::pow(lt::wharmonic_lt_with_derivative(wh, whd, v1), (int)sincIndex);
        T Wj     = hjInv3 * ::sphexa::math::pow(lt::wharmonic_lt_with_derivative(wh, whd, v2), (int)sincIndex);

        T termA1_i = c11i * rx + c12i * ry + c13i * rz;
        T termA2_i = c12i * rx + c22i * ry + c23i * rz;
        T termA3_i = c13i * rx + c23i * ry + c33i * rz;

        T c11j = c11[j];
        T c12j = c12[j];
        T c13j = c13[j];
        T c22j = c22[j];
        T c23j = c23[j];
        T c33j = c33[j];

        T termA1_j = c11j * rx + c12j * ry + c13j * rz;
        T termA2_j = c12j * rx + c22j * ry + c23j * rz;
        T termA3_j = c13j * rx + c23j * ry + c33j * rz;

        T roj = ro[j];
        T cj  = c[j];

        T wij = rv / dist;
        T viscosity_ij = T(0.5) * artificial_viscosity(ci, cj, wij);

        // For time-step calculations
        T vijsignal = ci + cj - 3.0 * wij;
        maxvsignali = (vijsignal > maxvsignali) ? vijsignal : maxvsignali;

        T mj        = m[j];
        T mj_roj_Wj = mj / roj * Wj;

        T mj_pro_i = mj * pri  / (gradh_i * roi * roi);

        {
            T a = Wi * (mj_pro_i + viscosity_ij * mi_roi);
            T b = mj_roj_Wj * (p[j] / (roj * gradh_j) + viscosity_ij);

            momentum_x += a * termA1_i + b * termA1_j;
            momentum_y += a * termA2_i + b * termA2_j;
            momentum_z += a * termA3_i + b * termA3_j;
        }
        {
            T a = Wi * (2.0 * mj_pro_i + viscosity_ij * mi_roi);
            T b = viscosity_ij * mj_roj_Wj;

            energy += vx_ij * (a * termA1_i + b * termA1_j) + vy_ij * (a * termA2_i + b * termA2_j) +
                      vz_ij * (a * termA3_i + b * termA3_j);
        }
    }

    // with the choice of calculating coordinate (r) and velocity (v_ij) differences as i - j,
    // we add the negative sign only here at the end instead of to termA123_ij in each iteration
    du[i]         = -K * 0.5 * energy;
    grad_P_x[i]   = -K * momentum_x;
    grad_P_y[i]   = -K * momentum_y;
    grad_P_z[i]   = -K * momentum_z;
    maxvsignal[i] = maxvsignali;
}

} // namespace kernels
} // namespace sph
} // namespace sphexa
