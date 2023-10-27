#pragma once

#include "cstone/sfc/box.hpp"

#include "sph/kernels.hpp"
#include "sph/table_lookup.hpp"

namespace sph
{

template<size_t stride = 1, class Tc, class Tm, class T>
HOST_DEVICE_FUN inline T densityJLoop(cstone::LocalIndex i, Tc K, const cstone::Box<Tc>& box,
                                      const cstone::LocalIndex* neighbors, unsigned neighborsCount, const Tc* x,
                                      const Tc* y, const Tc* z, const T* h, const Tm* m, const T* wh, const T* /*whd*/)
{
    auto xi = x[i];
    auto yi = y[i];
    auto zi = z[i];
    auto hi = h[i];

    T hInv  = T(1) / hi;
    T h3Inv = hInv * hInv * hInv;

    T roloc = 0.0;
    for (unsigned pj = 0; pj < neighborsCount; ++pj)
    {
        cstone::LocalIndex j = neighbors[stride * pj];

        T dist = distancePBC(box, hi, xi, yi, zi, x[j], y[j], z[j]);
        T vloc = dist * hInv;
        T w    = lt::lookup(wh, vloc);

        roloc += w * m[j];
    }

    return K * (roloc + m[i]) * h3Inv;
}

} // namespace sph
