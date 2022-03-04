#pragma once

#include "cstone/sfc/box.hpp"

#include "sph/lookupTables.hpp"

namespace sphexa
{
namespace sph
{
namespace kernels
{

template<typename T>
CUDA_DEVICE_HOST_FUN inline void IADJLoop(int i, T sincIndex, T K, const cstone::Box<T>& box, const int* neighbors,
                                          int neighborsCount, const T* x, const T* y, const T* z, const T* h, const T* m,
                                          const T* wh, const T* whd, const T* rho0, const T* kx,
                                          T* c11, T* c12, T* c13, T* c22, T* c23, T* c33)
{
    T tau11 = 0.0, tau12 = 0.0, tau13 = 0.0, tau22 = 0.0, tau23 = 0.0, tau33 = 0.0;

    T xi = x[i];
    T yi = y[i];
    T zi = z[i];

    T hi    = h[i];
    T hiInv = 1.0 / hi;
    T norm  = K * hiInv * hiInv * hiInv;

    for (int pj = 0; pj < neighborsCount; ++pj)
    {
        int j = neighbors[pj];

        T rx = (xi - x[j]);
        T ry = (yi - y[j]);
        T rz = (zi - z[j]);

        applyPBC(box, 2.0 * hi, rx, ry, rz);

        T dist = std::sqrt(rx * rx + ry * ry + rz * rz);

        // calculate the v as ratio between the distance and the smoothing length
        T vloc = dist * hiInv;
        T w = ::sphexa::math::pow(lt::wharmonic_lt_with_derivative(wh, whd, vloc), (int)sincIndex);

        T volj_w = m[j] / rho0[j] / kx[j] * w;

        tau11 += rx * rx * volj_w;
        tau12 += rx * ry * volj_w;
        tau13 += rx * rz * volj_w;
        tau22 += ry * ry * volj_w;
        tau23 += ry * rz * volj_w;
        tau33 += rz * rz * volj_w;
    }

    T det = tau11 * tau22 * tau33 + 2.0 * tau12 * tau23 * tau13 - tau11 * tau23 * tau23 - tau22 * tau13 * tau13 -
            tau33 * tau12 * tau12;

    // note normalization factor: cij have units of 1/tau because det is proportional to tau^3 so we have to
    // divide by K/h^3
    // taus should be multiplied by norm, so det should be norm**3
    // but we are interested on cXX vectors which are tau**2/det
    // That is 1/norm, so it is enough to multiply det by norm and
    // that gives cxx/norm.
    T factor = 1.0 / (det * norm);

    c11[i] = (tau22 * tau33 - tau23 * tau23) * factor;
    c12[i] = (tau13 * tau23 - tau33 * tau12) * factor;
    c13[i] = (tau12 * tau23 - tau22 * tau13) * factor;
    c22[i] = (tau11 * tau33 - tau13 * tau13) * factor;
    c23[i] = (tau13 * tau12 - tau11 * tau23) * factor;
    c33[i] = (tau11 * tau22 - tau12 * tau12) * factor;
}

} // namespace kernels
} // namespace sph
} // namespace sphexa