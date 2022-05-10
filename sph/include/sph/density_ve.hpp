#pragma once

#include <vector>

#include "cstone/findneighbors.hpp"

#include "sph/kernel_ve/density_kern.hpp"
#ifdef USE_CUDA
#include "cuda/sph.cuh"
#endif

namespace sphexa
{
namespace sph
{
template<class T, class Dataset>
void computeDensityVeImpl(size_t startIndex, size_t endIndex, size_t ngmax, Dataset& d, const cstone::Box<T>& box)
{
    const int* neighbors      = d.neighbors.data();
    const int* neighborsCount = d.neighborsCount.data();

    const T* x = d.x.data();
    const T* y = d.y.data();
    const T* z = d.z.data();
    const T* h = d.h.data();
    const T* m = d.m.data();

    const T* wh  = d.wh.data();
    const T* whd = d.whd.data();

    const T* xm = d.xm.data();

    T* kx    = d.kx.data();
    T* gradh = d.gradh.data();

    const T K         = d.K;
    const T sincIndex = d.sincIndex;

#pragma omp parallel for
    for (size_t i = startIndex; i < endIndex; i++)
    {
        size_t ni          = i - startIndex;
        auto [kxi, gradhi] = kernels::densityJLoop(
            i, sincIndex, K, box, neighbors + ngmax * ni, neighborsCount[i], x, y, z, h, m, wh, whd, xm);

        kx[i]    = kxi;
        gradh[i] = gradhi;

#ifndef NDEBUG
        T rhoi = kx[i] * m[i] / xm[i];
        if (std::isnan(rhoi))
            printf("ERROR::Density(%zu) density %f, position: (%f %f %f), h: %f\n", i, rhoi, x[i], y[i], z[i], h[i]);
#endif
    }
}

template<typename T, class Dataset>
void computeDensityVE(size_t startIndex, size_t endIndex, size_t ngmax, Dataset& d, const cstone::Box<T>& box)
{
    computeDensityVeImpl(startIndex, endIndex, ngmax, d, box);
}

} // namespace sph
} // namespace sphexa
