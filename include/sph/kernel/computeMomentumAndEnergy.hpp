#pragma once

#include "../lookupTables.hpp"

namespace sphexa
{
namespace sph
{
namespace kernels
{

template <typename T>
CUDA_DEVICE_HOST_FUN inline
void momentumAndEnergyJLoop(int pi, const T sincIndex, const T K, const int ngmax, const BBox<T>& bbox,
                                const int *clist, const int *neighbors, const int *neighborsCount, const T *x, const T *y,
                                const T *z, const T *vx, const T *vy, const T *vz, const T *h, const T *m, const T *ro,
                                const T *p, const T *c, const T *c11, const T *c12, const T *c13, const T *c22, const T *c23,
                                const T *c33, const T *wh, const T *whd, T *grad_P_x, T *grad_P_y, T *grad_P_z, T *du, T *maxvsignal)
{
    const T gradh_i = 1.0;
    const T gradh_j = 1.0;

    const int i = clist[pi];
    const int nn = neighborsCount[pi];

    T maxvsignali = 0.0;
    T momentum_x = 0.0, momentum_y = 0.0, momentum_z = 0.0, energy = 0.0, energyAV = 0.0;
    for (int pj = 0; pj < nn; ++pj)
    {
        const int j = neighbors[pi * ngmax + pj];

        T r_ijx = (x[i] - x[j]);
        T r_ijy = (y[i] - y[j]);
        T r_ijz = (z[i] - z[j]);

        T r_jix = (x[j] - x[i]);
        T r_jiy = (y[j] - y[i]);
        T r_jiz = (z[j] - z[i]);

        applyPBC(bbox, 2.0 * h[i], r_ijx, r_ijy, r_ijz);
        applyPBC(bbox, 2.0 * h[i], r_jix, r_jiy, r_jiz);

        const T dist = std::sqrt(r_ijx * r_ijx + r_ijy * r_ijy + r_ijz * r_ijz);

        const T v_ijx = (vx[i] - vx[j]);
        const T v_ijy = (vy[i] - vy[j]);
        const T v_ijz = (vz[i] - vz[j]);

        const T v1 = dist / h[i];
        const T v2 = dist / h[j];

        const T rv = r_ijx * v_ijx + r_ijy * v_ijy + r_ijz * v_ijz;

        const T w1 = K * ::sphexa::math::pow(lt::wharmonic_lt_with_derivative(wh, whd, v1), (int)sincIndex);
        const T w2 = K * ::sphexa::math::pow(lt::wharmonic_lt_with_derivative(wh, whd, v2), (int)sincIndex);

        const T W1 = w1 / (h[i] * h[i] * h[i]);
        const T W2 = w2 / (h[j] * h[j] * h[j]);

        const T kern11_i = c11[i] * r_jix;
        const T kern12_i = c12[i] * r_jiy;
        const T kern13_i = c13[i] * r_jiz;
        const T kern21_i = c12[i] * r_jix;
        const T kern22_i = c22[i] * r_jiy;
        const T kern23_i = c23[i] * r_jiz;
        const T kern31_i = c13[i] * r_jix;
        const T kern32_i = c23[i] * r_jiy;
        const T kern33_i = c33[i] * r_jiz;

        const T kern11_j = c11[j] * r_jix;
        const T kern12_j = c12[j] * r_jiy;
        const T kern13_j = c13[j] * r_jiz;
        const T kern21_j = c12[j] * r_jix;
        const T kern22_j = c22[j] * r_jiy;
        const T kern23_j = c23[j] * r_jiz;
        const T kern31_j = c13[j] * r_jix;
        const T kern32_j = c23[j] * r_jiy;
        const T kern33_j = c33[j] * r_jiz;

        const T termA1_i = (kern11_i + kern12_i + kern13_i) * W1;
        const T termA2_i = (kern21_i + kern22_i + kern23_i) * W1;
        const T termA3_i = (kern31_i + kern32_i + kern33_i) * W1;

        const T termA1_j = (kern11_j + kern12_j + kern13_j) * W2;
        const T termA2_j = (kern21_j + kern22_j + kern23_j) * W2;
        const T termA3_j = (kern31_j + kern32_j + kern33_j) * W2;

        const T pro_i = p[i] / (gradh_i * ro[i] * ro[i]);
        const T pro_j = p[j] / (gradh_j * ro[j] * ro[j]);

        const T r_square = dist * dist;
        const T viscosity_ij = artificial_viscosity(ro[i], ro[j], h[i], h[j], c[i], c[j], rv, r_square);

        // For time-step calculations
        const T wij = rv / dist;
        const T vijsignal = c[i] + c[j] - 3.0 * wij;
        if (vijsignal > maxvsignali) maxvsignali = vijsignal;

        const T grad_Px_AV = 0.5 * (m[i] / ro[i] * viscosity_ij * termA1_i + m[j] / ro[j] * viscosity_ij * termA1_j);
        const T grad_Py_AV = 0.5 * (m[i] / ro[i] * viscosity_ij * termA2_i + m[j] / ro[j] * viscosity_ij * termA2_j);
        const T grad_Pz_AV = 0.5 * (m[i] / ro[i] * viscosity_ij * termA3_i + m[j] / ro[j] * viscosity_ij * termA3_j);

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
