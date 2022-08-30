#pragma once

#include "cstone/sfc/box.hpp"

#include "sph/kernels.hpp"
#include "sph/math.hpp"
#include "sph/tables.hpp"

namespace sph
{

template<typename T>
HOST_DEVICE_FUN inline T densityJLoop(cstone::LocalIndex i, T sincIndex, T K, const cstone::Box<T>& box,
                                      const cstone::LocalIndex* neighbors, unsigned neighborsCount, const T* x,
                                      const T* y, const T* z, const T* h, const T* m, const T* wh, const T* whd)
{
    T xi = x[i];
    T yi = y[i];
    T zi = z[i];
    T hi = h[i];

    T hInv  = 1.0 / hi;
    T h3Inv = hInv * hInv * hInv;

    T roloc = 0.0;
    for (unsigned pj = 0; pj < neighborsCount; ++pj)
    {
        cstone::LocalIndex j = neighbors[pj];

        T dist = distancePBC(box, hi, xi, yi, zi, x[j], y[j], z[j]);
        T vloc = dist * hInv;
        T w    = math::pow(lt::wharmonic_lt_with_derivative(wh, whd, vloc), (int)sincIndex);

        roloc += w * m[j];
    }

    return K * (roloc + m[i]) * h3Inv;
}

} // namespace sph
