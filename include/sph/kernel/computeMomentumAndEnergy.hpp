#pragma once

#include "../lookupTables.hpp"

namespace sphexa
{
namespace sph
{
namespace kernels
{

template<typename T>
CUDA_DEVICE_HOST_FUN inline void
momentumAndEnergyJLoop(int pi, const T sincIndex, const T K, const int ngmax, const BBox<T>& bbox, const int* clist,
                       const int* neighbors, const int* neighborsCount, const T* x, const T* y, const T* z, const T* vx,
                       const T* vy, const T* vz, const T* h, const T* m, const T* ro, const T* p, const T* c,
                       const T* c11, const T* c12, const T* c13, const T* c22, const T* c23, const T* c33, const T* wh,
                       const T* whd, T* grad_P_x, T* grad_P_y, T* grad_P_z, T* du, T* maxvsignal)
{
    constexpr T gradh_i = 1.0;
    constexpr T gradh_j = 1.0;

    int i = clist[pi];
    int nn = neighborsCount[pi];

    T xi = x[i];
    T yi = y[i];
    T zi = z[i];
    T vxi = vx[i];
    T vyi = vy[i];
    T vzi = vz[i];

    T hi  = h[i];
    T roi = ro[i];
    T pri = p[i];
    T ci  = c[i];

    T mi_roi  = m[i] / ro[i];

    T hiInv = 1.0 / hi;
    T hiInv3 = hiInv * hiInv * hiInv;

    T maxvsignali = 0.0;
    T momentum_x = 0.0, momentum_y = 0.0, momentum_z = 0.0, energy = 0.0;

    T c11i = c11[i];
    T c12i = c12[i];
    T c13i = c13[i];
    T c22i = c22[i];
    T c23i = c23[i];
    T c33i = c33[i];

    for (int pj = 0; pj < nn; ++pj)
    {
        int j = neighbors[pi * ngmax + pj];

        T rx    = x[j] - xi;
        T ry    = y[j] - yi;
        T rz    = z[j] - zi;

        applyPBC(bbox, 2.0 * hi, rx, ry, rz);

        T r2   = rx * rx + ry * ry + rz * rz;
        T dist = std::sqrt(r2);

        T vx_ji = vx[j] - vxi;
        T vy_ji = vy[j] - vyi;
        T vz_ji = vz[j] - vzi;

        T hj    = h[j];
        T hjInv = 1.0 / hj;

        T v1 = dist * hiInv;
        T v2 = dist * hjInv;

        T rv = rx * vx_ji + ry * vy_ji + rz * vz_ji;

        T hjInv3 = hjInv * hjInv * hjInv;
        T W1 = hiInv3 * ::sphexa::math::pow(lt::wharmonic_lt_with_derivative(wh, whd, v1), (int)sincIndex);
        T W2 = hjInv3 * ::sphexa::math::pow(lt::wharmonic_lt_with_derivative(wh, whd, v2), (int)sincIndex);

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

        T viscosity_ij = T(0.5) * artificial_viscosity(roi, roj, hi, hj, ci, cj, rv, r2);

        // For time-step calculations
        T wij = rv / dist;
        T vijsignal = ci + cj - 3.0 * wij;

        if (vijsignal > maxvsignali)
        {
            maxvsignali = vijsignal;
        }

        T mj     = m[j];
        T mj_roj = W2 * mj / roj;

        T pro_i = mj * pri  / (gradh_i * roi * roi);

        {
            T a = W1 * (pro_i + viscosity_ij * mi_roi);
            T b = mj_roj * (p[j] / (roj * gradh_j) + viscosity_ij);

            momentum_x += a * termA1_i + b * termA1_j;
            momentum_y += a * termA2_i + b * termA2_j;
            momentum_z += a * termA3_i + b * termA3_j;
        }
        {
            T a = W1 * (2.0 * pro_i + viscosity_ij * mi_roi);
            T b = viscosity_ij * mj_roj;

            energy += vx_ji * (a * termA1_i + b * termA1_j)
                    + vy_ji * (a * termA2_i + b * termA2_j)
                    + vz_ji * (a * termA3_i + b * termA3_j);
        }
    }

    du[i]         = -K * 0.5 * energy;
    grad_P_x[i]   = K * momentum_x;
    grad_P_y[i]   = K * momentum_y;
    grad_P_z[i]   = K * momentum_z;
    maxvsignal[i] = maxvsignali;
}

} // namespace kernels
} // namespace sph
} // namespace sphexa
