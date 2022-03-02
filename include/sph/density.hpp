#pragma once

#include <vector>

#include "cstone/findneighbors.hpp"

#include "kernel/density.hpp"
#ifdef USE_CUDA
#include "cuda/sph.cuh"
#endif

namespace sphexa
{
namespace sph
{
template<class T, class Dataset>
void computeDensityImpl(size_t startIndex, size_t endIndex, size_t ngmax, Dataset& d, const cstone::Box<T>& box)
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

    T* rho = d.rho.data();

    const T K         = d.K;
    const T sincIndex = d.sincIndex;

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; i++)
    {
        // int neighLoc[ngmax];
        // int count;
        // cstone::findNeighbors(
        //    pi, x, y, z, h, box, cstone::sfcKindPointer(d.codes.data()), neighLoc, &count, d.codes.size(), ngmax);

        size_t ni = i - startIndex;

        rho[i] = kernels::densityJLoop(
            i, sincIndex, K, box, neighbors + ngmax * ni, neighborsCount[i], x, y, z, h, m, wh, whd);

#ifndef NDEBUG
        if (std::isnan(rho[i]))
            printf("ERROR::Density(%zu) density %f, position: (%f %f %f), h: %f\n", i, rho[i], x[i], y[i], z[i], h[i]);
#endif
    }
}

template<class T, class Dataset>
void computeDensity(size_t startIndex, size_t endIndex, size_t ngmax, Dataset& d, const cstone::Box<T>& box)
{
#if defined(USE_CUDA)
    cuda::computeDensity(startIndex, endIndex, ngmax, d, box);
#else
    computeDensityImpl(startIndex, endIndex, ngmax, d, box);
#endif
}

} // namespace sph
} // namespace sphexa
