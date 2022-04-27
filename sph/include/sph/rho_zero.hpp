#pragma once

#include <vector>

#include "cstone/findneighbors.hpp"

#include "kernels.hpp"
#include "kernel/rho_zero_kern.hpp"

#ifdef USE_CUDA
#include "cuda/sph.cuh"
#endif

namespace sphexa
{
namespace sph
{
template<typename T, class Dataset>
void computeRhoZeroImpl(size_t startIndex, size_t endIndex, size_t ngmax, Dataset& d, const cstone::Box<T>& box)
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

    T* rho0  = d.rho0.data();
    T* wrho0 = d.wrho0.data();

    const T K         = d.K;
    const T sincIndex = d.sincIndex;

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; i++)
    {
        size_t ni = i - startIndex;

        kernels::rhoZeroJLoop(
            i, sincIndex, K, box, neighbors + ngmax * ni, neighborsCount[i], x, y, z, h, m, wh, whd, rho0, wrho0);

#ifndef NDEBUG
        if (std::isnan(rho0[i]))
            printf("ERROR::Rho0(%zu) rho0 %f, position: (%f %f %f), h: %f\n", i, rho0[i], x[i], y[i], z[i], h[i]);
#endif
    }
}

template<typename T, class Dataset>
void computeRhoZero(size_t startIndex, size_t endIndex, size_t ngmax, Dataset& d, const cstone::Box<T>& box)
{
#if defined(USE_CUDA)
    cuda::computeRhoZero(startIndex, endIndex, ngmax, d, box);
#else
    computeRhoZeroImpl(startIndex, endIndex, ngmax, d, box);
#endif
}

} // namespace sph
} // namespace sphexa
