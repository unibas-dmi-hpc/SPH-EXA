#pragma once

#include "BBox.hpp"
#include "sph/lookupTables.hpp"

namespace sphexa
{
namespace sph
{
namespace kernels
{

template<typename T>
CUDA_DEVICE_HOST_FUN inline void densityJLoop(int pi, T sincIndex, T K, int ngmax, const BBox<T>& bbox,
                                              const int* clist, const int* neighbors, const int* neighborsCount,
                                              const T* x, const T* y, const T* z, const T* h, const T* m, const T* wh,
                                              const T* whd, T* ro)
{
    const int i  = clist[pi];
    const int nn = neighborsCount[pi];

    T xi = x[i];
    T yi = y[i];
    T zi = z[i];
    T hi = h[i];

    T hInv      = 1.0 / hi;
    T volumeInv = hInv * hInv * hInv;

    const int* neighborsOfI = neighbors + pi * ngmax;

    T roloc = 0.0;
    for (int pj = 0; pj < nn; ++pj)
    {
        int j  = neighborsOfI[pj];
        T dist = distancePBC(bbox, hi, xi, yi, zi, x[j], y[j], z[j]);
        T vloc = dist * hInv;
        T w    = ::sphexa::math::pow(lt::wharmonic_lt_with_derivative(wh, whd, vloc), (int)sincIndex);

        roloc += w * m[j];
    }

    ro[i] = K * (roloc + m[i]) * volumeInv;
}

} // namespace kernels
} // namespace sph
} // namespace sphexa
