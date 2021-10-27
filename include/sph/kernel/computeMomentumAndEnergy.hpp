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
    const T gradh_i = 1.0;
    const T gradh_j = 1.0;

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

    T maxvsignali = 0.0;
    T momentum_x = 0.0, momentum_y = 0.0, momentum_z = 0.0, energy = 0.0, energyAV = 0.0;

    for (int pj = 0; pj < nn; ++pj)
    {
        const int j = neighbors[pi * ngmax + pj];

        T rx    = (x[j] - xi);
        T ry    = (y[j] - yi);
        T rz    = (z[j] - zi);

        applyPBC(bbox, 2.0 * hi, rx, ry, rz);

        const T dist = std::sqrt(rx * rx + ry * ry + rz * rz);

        const T v_ijx = (vxi - vx[j]);
        const T v_ijy = (vyi - vy[j]);
        const T v_ijz = (vzi - vz[j]);

        const T v1 = dist / hi;
        const T v2 = dist / h[j];

        const T rv = -(rx * v_ijx + ry * v_ijy + rz * v_ijz);

        const T w1 = K * ::sphexa::math::pow(lt::wharmonic_lt_with_derivative(wh, whd, v1), (int)sincIndex);
        const T w2 = K * ::sphexa::math::pow(lt::wharmonic_lt_with_derivative(wh, whd, v2), (int)sincIndex);

        const T W1 = w1 / (hi * hi * hi);
        const T W2 = w2 / (h[j] * h[j] * h[j]);

        const T kern11_i = c11[i] * rx;
        const T kern12_i = c12[i] * ry;
        const T kern13_i = c13[i] * rz;
        const T kern21_i = c12[i] * rx;
        const T kern22_i = c22[i] * ry;
        const T kern23_i = c23[i] * rz;
        const T kern31_i = c13[i] * rx;
        const T kern32_i = c23[i] * ry;
        const T kern33_i = c33[i] * rz;

        const T kern11_j = c11[j] * rx;
        const T kern12_j = c12[j] * ry;
        const T kern13_j = c13[j] * rz;
        const T kern21_j = c12[j] * rx;
        const T kern22_j = c22[j] * ry;
        const T kern23_j = c23[j] * rz;
        const T kern31_j = c13[j] * rx;
        const T kern32_j = c23[j] * ry;
        const T kern33_j = c33[j] * rz;

        const T termA1_i = (kern11_i + kern12_i + kern13_i) * W1;
        const T termA2_i = (kern21_i + kern22_i + kern23_i) * W1;
        const T termA3_i = (kern31_i + kern32_i + kern33_i) * W1;

        const T termA1_j = (kern11_j + kern12_j + kern13_j) * W2;
        const T termA2_j = (kern21_j + kern22_j + kern23_j) * W2;
        const T termA3_j = (kern31_j + kern32_j + kern33_j) * W2;

        const T pro_i = pri  / (gradh_i * roi * roi);
        const T pro_j = p[j] / (gradh_j * ro[j] * ro[j]);

        const T r_square = dist * dist;
        const T viscosity_ij = artificial_viscosity(roi, ro[j], hi, h[j], ci, c[j], rv, r_square);

        // For time-step calculations
        const T wij = rv / dist;
        const T vijsignal = ci + c[j] - 3.0 * wij;
        if (vijsignal > maxvsignali) maxvsignali = vijsignal;

        T mj_roj = m[j] / ro[j];
        const T grad_Px_AV = 0.5 * viscosity_ij * (mi_roi * termA1_i + mj_roj * termA1_j);
        const T grad_Py_AV = 0.5 * viscosity_ij * (mi_roi * termA2_i + mj_roj * termA2_j);
        const T grad_Pz_AV = 0.5 * viscosity_ij * (mi_roi * termA3_i + mj_roj * termA3_j);

        momentum_x += m[j] * (pro_i * termA1_i + pro_j * termA1_j) + grad_Px_AV;
        momentum_y += m[j] * (pro_i * termA2_i + pro_j * termA2_j) + grad_Py_AV;
        momentum_z += m[j] * (pro_i * termA3_i + pro_j * termA3_j) + grad_Pz_AV;

        energy += m[j] * 2.0 * pro_i * (v_ijx * termA1_i + v_ijy * termA2_i + v_ijz * termA3_i);
        energyAV += grad_Px_AV * v_ijx + grad_Py_AV * v_ijy + grad_Pz_AV * v_ijz;
    }

    du[i] = 0.5 * (energy + energyAV);
    grad_P_x[i] = momentum_x;
    grad_P_y[i] = momentum_y;
    grad_P_z[i] = momentum_z;
    maxvsignal[i] = maxvsignali;
}

} // namespace kernels
} // namespace sph
} // namespace sphexa
