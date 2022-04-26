#pragma once

#include "kernel/divv_curlv_kern.hpp"

#ifdef USE_CUDA
#include "cuda/sph.cuh"
#endif

namespace sphexa
{
namespace sph
{

template<class T, class Dataset>
void computeDivvCurlvImpl(size_t startIndex, size_t endIndex, size_t ngmax, Dataset& d, const cstone::Box<T>& box)
{
    const int* neighbors      = d.neighbors.data();
    const int* neighborsCount = d.neighborsCount.data();

    const T* h  = d.h.data();
    const T* m  = d.m.data();
    const T* x  = d.x.data();
    const T* y  = d.y.data();
    const T* z  = d.z.data();
    const T* vx = d.vx.data();
    const T* vy = d.vy.data();
    const T* vz = d.vz.data();

    const T* c11 = d.c11.data();
    const T* c12 = d.c12.data();
    const T* c13 = d.c13.data();
    const T* c22 = d.c22.data();
    const T* c23 = d.c23.data();
    const T* c33 = d.c33.data();

    T* divv  = d.divv.data();
    T* curlv = d.curlv.data();

    const T* wh   = d.wh.data();
    const T* whd  = d.whd.data();
    const T* kx   = d.kx.data();
    const T* rho0 = d.rho0.data();

    const T K         = d.K;
    const T sincIndex = d.sincIndex;

#pragma omp parallel for
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        size_t ni = i - startIndex;
        kernels::divV_curlVJLoop(i,
                                 sincIndex,
                                 K,
                                 box,
                                 neighbors + ngmax * ni,
                                 neighborsCount[ni],
                                 x,
                                 y,
                                 z,
                                 vx,
                                 vy,
                                 vz,
                                 h,
                                 m,
                                 c11,
                                 c12,
                                 c13,
                                 c22,
                                 c23,
                                 c33,
                                 wh,
                                 whd,
                                 kx,
                                 rho0,
                                 divv,
                                 curlv);
    }
}

template<class T, class Dataset>
void computeDivvCurlv(size_t startIndex, size_t endIndex, size_t ngmax, Dataset& d, const cstone::Box<T>& box)
{
#if defined(USE_CUDA)
    cuda::computeDivvCurlv(startIndex, endIndex, ngmax, d, box);
#else
    computeDivvCurlvImpl(startIndex, endIndex, ngmax, d, box);
#endif
}

} // namespace sph
} // namespace sphexa
