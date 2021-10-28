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
    T momentum_x = 0.0, momentum_y = 0.0, momentum_z = 0.0, energy = 0.0, energyAV = 0.0;

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

        T termA1_i = (c11i * rx + c12i * ry + c13i * rz) * W1;
        T termA2_i = (c12i * rx + c22i * ry + c23i * rz) * W1;
        T termA3_i = (c13i * rx + c23i * ry + c33i * rz) * W1;

        T c11j = c11[j];
        T c12j = c12[j];
        T c13j = c13[j];
        T c22j = c22[j];
        T c23j = c23[j];
        T c33j = c33[j];

        T termA1_j = (c11j * rx + c12j * ry + c13j * rz) * W2;
        T termA2_j = (c12j * rx + c22j * ry + c23j * rz) * W2;
        T termA3_j = (c13j * rx + c23j * ry + c33j * rz) * W2;

        T roj = ro[j];
        T cj  = c[j];

        T viscosity_ij = artificial_viscosity(roi, roj, hi, hj, ci, cj, rv, r2);

        // For time-step calculations
        T wij = rv / dist;
        T vijsignal = ci + cj - 3.0 * wij;

        if (vijsignal > maxvsignali)
        {
            maxvsignali = vijsignal;
        }

        T mj     = m[j];
        T mj_roj = mj / roj;

        T grad_Px_AV = T(0.5) * viscosity_ij * (mi_roi * termA1_i + mj_roj * termA1_j);
        T grad_Py_AV = T(0.5) * viscosity_ij * (mi_roi * termA2_i + mj_roj * termA2_j);
        T grad_Pz_AV = T(0.5) * viscosity_ij * (mi_roi * termA3_i + mj_roj * termA3_j);

        T pro_i = mj * pri  / (gradh_i * roi * roi);
        T pro_j = mj * p[j] / (gradh_j * roj * roj);

        momentum_x += pro_i * termA1_i + pro_j * termA1_j + grad_Px_AV;
        momentum_y += pro_i * termA2_i + pro_j * termA2_j + grad_Py_AV;
        momentum_z += pro_i * termA3_i + pro_j * termA3_j + grad_Pz_AV;

        energy   += 2.0 * pro_i * (vx_ji * termA1_i + vy_ji * termA2_i + vz_ji * termA3_i);
        energyAV += grad_Px_AV * vx_ji + grad_Py_AV * vy_ji + grad_Pz_AV * vz_ji;
    }

    du[i]         = -K * 0.5 * (energy + energyAV);
    grad_P_x[i]   = K * momentum_x;
    grad_P_y[i]   = K * momentum_y;
    grad_P_z[i]   = K * momentum_z;
    maxvsignal[i] = maxvsignali;
}

} // namespace kernels
} // namespace sph
} // namespace sphexa
