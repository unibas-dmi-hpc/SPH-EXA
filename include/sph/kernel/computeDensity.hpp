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
void densityJLoop(int pi, T sincIndex, T K, int ngmax, const BBox<T> *bbox, const int *clist, const int *neighbors, const int *neighborsCount,
                  const T *x, const T *y, const T *z, const T *h, const T *m, const T *wh, const T *whd, size_t ltsize, T *ro)
{
    const int i = clist[pi];
    const int nn = neighborsCount[pi];

    T roloc = 0.0;
    for (int pj = 0; pj < nn; ++pj)
    {
        const int j = neighbors[pi * ngmax + pj];
        const T dist = distancePBC(*bbox, h[i], x[i], y[i], z[i], x[j], y[j], z[j]);
        const T vloc = dist / h[i];
        const T w = K * math_namespace::pow(lt::wharmonic_lt_with_derivative(wh, whd, ltsize, vloc), (int)sincIndex);
        const T value = w / (h[i] * h[i] * h[i]);
        roloc += value * m[j];
    }

    ro[i] = roloc + m[i] * K / (h[i] * h[i] * h[i]);
}

} // namespace kernels
} // namespace sph
} // namespace sphexa
