#pragma once

#include "BBox.hpp"

namespace sphexa
{
namespace sph
{
namespace kernels
{

template <typename T>
#if defined(USE_CUDA)
//__global__
#endif
void density(int pi, const int n, const T sincIndex, const T K, const int ngmax, const BBox<T> *bbox, const int *clist,
                        const int *neighbors, const int *neighborsCount, const T *x, const T *y, const T *z, const T *h, const T *m, const T *wh, const T *whd, const size_t ltsize, T *ro)
{
#if defined(USE_CUDA)
    //pi = blockDim.x * blockIdx.x + threadIdx.x;
    //if (pi >= n) return;
#endif

    const int i = clist[pi];
    const int nn = neighborsCount[pi];

    T roloc = 0.0;

    // int converstion to avoid a bug that prevents vectorization with some compilers
    for (int pj = 0; pj < nn; pj++)
    {
        const int j = neighbors[pi * ngmax + pj];

        // later can be stores into an array per particle
        T dist = distancePBC<T>(*bbox, h[i], x[i], y[i], z[i], x[j], y[j], z[j]); // store the distance from each neighbor

        // calculate the v as ratio between the distance and the smoothing length
        T vloc = dist / h[i];

#ifndef NDEBUG
        if (vloc > 2.0 + 1e-6 || vloc < 0.0)
            printf("ERROR:Density(%d,%d) vloc %f -- x %f %f %f -- %f %f %f -- dist %f -- hi %f\n", i, j, vloc, x[i], y[i], z[i], x[j],
                    y[j], z[j], dist, h[i]);
#endif

        const T w = K * math_namespace::pow(lt::wharmonic_lt_with_derivative(wh, whd, ltsize, vloc), (int)sincIndex);
        const T value = w / (h[i] * h[i] * h[i]);
        roloc += value * m[j];
    }

    ro[i] = roloc + m[i] * K / (h[i] * h[i] * h[i]);

}

} // namespace kernels
} // namespace sph
} // namespace sphexa
