#pragma once

#include "cstone/sfc/box.hpp"

#include "sph/tables.hpp"

namespace sph
{

template<typename T>
CUDA_DEVICE_HOST_FUN inline T densityJLoop(int i, T sincIndex, T K, const cstone::Box<T>& box, const int* neighbors,
                                           int neighborsCount, const T* x, const T* y, const T* z, const T* h,
                                           const T* m, const T* wh, const T* whd)
{
    T xi = x[i];
    T yi = y[i];
    T zi = z[i];
    T hi = h[i];

    T hInv  = 1.0 / hi;
    T h3Inv = hInv * hInv * hInv;

    T roloc = 0.0;
    for (int pj = 0; pj < neighborsCount; ++pj)
    {
        int j    = neighbors[pj];
        T   dist = distancePBC(box, hi, xi, yi, zi, x[j], y[j], z[j]);
        T   vloc = dist * hInv;
        T   w    = math::pow(lt::wharmonic_lt_with_derivative(wh, whd, vloc), (int)sincIndex);

        roloc += w * m[j];
    }

    return K * (roloc + m[i]) * h3Inv;
}

} // namespace sph
