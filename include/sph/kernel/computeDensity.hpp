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
CUDA_DEVICE_HOST_FUN inline T densityJLoop(int i, T sincIndex, T K, const cstone::Box<T>& box, const int* neighbors,
                                           int neighborsCount, const T* x, const T* y, const T* z, const T* h,
                                           const T* m, const T* wh, const T* whd, const T* xmass, T rho0i, T wrho0i)
{
    T xi = x[i];
    T yi = y[i];
    T zi = z[i];
    T hi = h[i];
    T xmassi = xmass[i];

    T hInv  = 1.0 / hi;
    T h3Inv = hInv * hInv * hInv;

    kx[i] = 0.0;
    wh[i] = 0.0;
    for (int pj = 0; pj < neighborsCount; ++pj)
    {
        int j  = neighbors[pj];
        T dist = distancePBC(box, hi, xi, yi, zi, x[j], y[j], z[j]);
        T vloc = dist * hInv;
        T w    = ::sphexa::math::pow(lt::wharmonic_lt_with_derivative(wh, whd, vloc), (int)sincIndex);
        T dw    = wharmonic_derivative_std(vloc);
        T dterh = -(3.0 * w + vloc * dw);

        kx[i] += w * xmass[j];
        wh[i] += dterh * xmass[j];
    }

    kx[i] = K * (kx[i] + xmassi) * h3Inv;
    wh[i] = K * (wh[i] - 3.0 * xmassi) * h3Inv * hInv;

    T roloc = kx[i] * m[i] / xmassi;
    wh[i] = wh[i] * m[i] / xmassi + (roloc / rho0i - K * xmassi * h3Inv) * wrho0i;

    return roloc;
}

} // namespace kernels
} // namespace sph
} // namespace sphexa
