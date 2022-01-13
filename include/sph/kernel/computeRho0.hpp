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
CUDA_DEVICE_HOST_FUN inline T Rho0JLoop(int i, T sincIndex, T K, const cstone::Box<T>& box, const int* neighbors,
                                           int neighborsCount, const T* x, const T* y, const T* z, const T* h,
                                           const T* m, const T* wh, const T* whd, T* rho0, T* wrho0, T* xmass)
{
    T xi = x[i];
    T yi = y[i];
    T zi = z[i];
    T hi = h[i];

    T hInv  = 1.0 / hi;
    T h3Inv = hInv * hInv * hInv;

    T rho0[i] = 0.0;
    T wrho0[i] = 0.0;
    for (int pj = 0; pj < neighborsCount; ++pj)
    {
        int j   = neighbors[pj];
        T dist  = distancePBC(box, hi, xi, yi, zi, x[j], y[j], z[j]);
        T vloc  = dist * hInv;
        T w     = ::sphexa::math::pow(lt::wharmonic_lt_with_derivative(wh, whd, vloc), (int)sincIndex);
        T dw    = wharmonic_derivative_std(vloc);
        T dterh = -(3.0 * w + vloc * dw);

        rho0[i] += w * m[j];
        wrho0[i] += dterh * m[j];
    }

    //XXXX Define new arrays rho0, wrho0, xmass for all particles
    rho0[i] = K * (rho0[i] + m[i]) * h3Inv;
    wrho0[i] = (wrho0[i] - 3.0 * m[i]) * h3Inv * hInv;
    xmass[i] = m[i] / rho0[i];
}

} // namespace kernels
} // namespace sph
} // namespace sphexa
