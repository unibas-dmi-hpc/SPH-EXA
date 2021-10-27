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
void IADJLoop(int pi, T sincIndex, T K, int ngmax, const BBox<T>& bbox, const int* clist,
              const int* neighbors, const int* neighborsCount,
              const T* x, const T* y, const T* z, const T* h, const T* m,
              const T* ro, const T* wh, const T* whd, T* c11, T* c12, T* c13, T* c22, T* c23, T* c33)
{
    const int i = clist[pi];
    const int nn = neighborsCount[pi];

    T tau11 = 0.0, tau12 = 0.0, tau13 = 0.0, tau22 = 0.0, tau23 = 0.0, tau33 = 0.0;

    for (int pj = 0; pj < nn; ++pj)
    {
        const int j = neighbors[pi * ngmax + pj];

        // later can be stored into an array per particle
        const T dist = distancePBC(bbox, h[i], x[i], y[i], z[i], x[j], y[j], z[j]); // store the distance from each neighbor
        // calculate the v as ratio between the distance and the smoothing length
        const T vloc = dist / h[i];
        const T w = K * ::sphexa::math::pow(lt::wharmonic_lt_with_derivative(wh, whd, vloc), (int)sincIndex);
        const T W = w / (h[i] * h[i] * h[i]);

        T r_ijx = (x[i] - x[j]);
        T r_ijy = (y[i] - y[j]);
        T r_ijz = (z[i] - z[j]);

        applyPBC(bbox, 2.0 * h[i], r_ijx, r_ijy, r_ijz);

        tau11 += r_ijx * r_ijx * m[j] / ro[j] * W;
        tau12 += r_ijx * r_ijy * m[j] / ro[j] * W;
        tau13 += r_ijx * r_ijz * m[j] / ro[j] * W;
        tau22 += r_ijy * r_ijy * m[j] / ro[j] * W;
        tau23 += r_ijy * r_ijz * m[j] / ro[j] * W;
        tau33 += r_ijz * r_ijz * m[j] / ro[j] * W;
    }

    const T det =
        tau11 * tau22 * tau33 + 2.0 * tau12 * tau23 * tau13 - tau11 * tau23 * tau23 - tau22 * tau13 * tau13 - tau33 * tau12 * tau12;

    c11[i] = (tau22 * tau33 - tau23 * tau23) / det;
    c12[i] = (tau13 * tau23 - tau33 * tau12) / det;
    c13[i] = (tau12 * tau23 - tau22 * tau13) / det;
    c22[i] = (tau11 * tau33 - tau13 * tau13) / det;
    c23[i] = (tau13 * tau12 - tau11 * tau23) / det;
    c33[i] = (tau11 * tau22 - tau12 * tau12) / det;
}

} // namespace kernels
} // namespace sph
} // namespace sphexa
