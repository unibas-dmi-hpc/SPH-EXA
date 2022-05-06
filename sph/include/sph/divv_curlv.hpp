#pragma once

#include "kernel_ve/divv_curlv_kern.hpp"
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

    const T* x  = d.x.data();
    const T* y  = d.y.data();
    const T* z  = d.z.data();
    const T* vx = d.vx.data();
    const T* vy = d.vy.data();
    const T* vz = d.vz.data();
    const T* h  = d.h.data();

    const T* c11 = d.c11.data();
    const T* c12 = d.c12.data();
    const T* c13 = d.c13.data();
    const T* c22 = d.c22.data();
    const T* c23 = d.c23.data();
    const T* c33 = d.c33.data();

    T* divv  = d.divv.data();
    T* curlv = d.curlv.data();

    const T* wh  = d.wh.data();
    const T* whd = d.whd.data();
    const T* kx  = d.kx.data();
    const T* xm  = d.xm.data();

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
                                 neighborsCount[i],
                                 x,
                                 y,
                                 z,
                                 vx,
                                 vy,
                                 vz,
                                 h,
                                 c11,
                                 c12,
                                 c13,
                                 c22,
                                 c23,
                                 c33,
                                 wh,
                                 whd,
                                 kx,
                                 xm,
                                 divv,
                                 curlv);
    }
}

template<class T, class Dataset>
void computeDivvCurlv(size_t startIndex, size_t endIndex, size_t ngmax, Dataset& d, const cstone::Box<T>& box)
{
    computeDivvCurlvImpl(startIndex, endIndex, ngmax, d, box);
}

} // namespace sph
} // namespace sphexa
