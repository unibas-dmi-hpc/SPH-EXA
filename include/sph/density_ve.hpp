#pragma once

#include <vector>

#include "cstone/findneighbors.hpp"

#include "sph/kernel_ve/density.hpp"
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

    const T* h = d.h.data();
    const T* m = d.m.data();
    const T* x = d.x.data();
    const T* y = d.y.data();
    const T* z = d.z.data();

    const T* wh  = d.wh.data();
    const T* whd = d.whd.data();

    const T* rho0  = d.rho0.data();
    const T* wrho0 = d.wrho0.data();

    T* kx      = d.kx.data();
    T* whomega = d.whomega.data();
    T* rho     = d.rho.data();

    const T K         = d.K;
    const T sincIndex = d.sincIndex;

#pragma omp parallel for
    for (size_t i = startIndex; i < endIndex; i++)
    {
        size_t ni = i - startIndex;
        kernels::densityJLoop(i,
                              sincIndex,
                              K,
                              box,
                              neighbors + ngmax * ni,
                              neighborsCount[i],
                              x,
                              y,
                              z,
                              h,
                              m,
                              wh,
                              whd,
                              rho0,
                              wrho0,
                              rho,
                              kx,
                              whomega);
#ifndef NDEBUG
        if (std::isnan(rho[i]))
            printf("ERROR::Density(%zu) density %f, position: (%f %f %f), h: %f\n", i, rho[i], x[i], y[i], z[i], h[i]);
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
