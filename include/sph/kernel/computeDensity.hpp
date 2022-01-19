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
CUDA_DEVICE_HOST_FUN inline void densityJLoop(int i, T sincIndex, T K, const cstone::Box<T>& box, const int* neighbors,
                                           int neighborsCount, const T* x, const T* y, const T* z, const T* h,
                                           const T* m, const T* wh, const T* whd, const T* rho0, const T* wrho0, T* ro,
                                           T* kx, T* whomega)
{
    T xi = x[i];
    T yi = y[i];
    T zi = z[i];
    T hi = h[i];
    T rho0i  = rho0[i];
    T wrho0i = wrho0[i];
    T xmassi = m[i] / rho0i;

    T hInv  = 1.0 / hi;
    T h3Inv = hInv * hInv * hInv;

    T kxi      = 0.0;
    T whomegai = 0.0;

    for (int pj = 0; pj < neighborsCount; ++pj)
    {
        int j  = neighbors[pj];
        T dist = distancePBC(box, hi, xi, yi, zi, x[j], y[j], z[j]);
        T vloc = dist * hInv;
        T w    = ::sphexa::math::pow(lt::wharmonic_lt_with_derivative(wh, whd, vloc), sincIndex);
        T dw   = ::sphexa::math::pow(wharmonic_derivative_std(vloc), (int)sincIndex - 1) * sincIndex;
        T dterh  = -(3.0 * w + vloc * dw);
        T xmassj = m[j] / rho0[j];

        kxi      += w * xmassj;
        whomegai += dterh * xmassj;
    }

    kxi      = K * (kxi + xmassi) * h3Inv;
    whomegai = K * (whomegai - 3.0 * xmassi) * h3Inv * hInv;

    T roloc  = kxi * m[i] / xmassi;
    whomegai = whomegai * m[i] / xmassi + (roloc / rho0i - K * xmassi * h3Inv) * wrho0i;

    ro[i]      = roloc;
    kx[i]      = kxi;
    whomega[i] = whomegai;
}

} // namespace kernels
} // namespace sph
} // namespace sphexa
