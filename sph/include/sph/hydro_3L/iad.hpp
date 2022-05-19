#pragma once

#include "sph/math.hpp"
#include "sph/kernels.hpp"
#include "kernel/iad_kern.hpp"
#ifdef USE_CUDA
#include "sph/cuda/sph.cuh"
#endif

namespace sphexa
{
namespace sph
{

template<class T, class Dataset>
void computeIADImpl(size_t startIndex, size_t endIndex, size_t ngmax, Dataset& d, const cstone::Box<T>& box)
{
    const int* neighbors      = d.neighbors.data();
    const int* neighborsCount = d.neighborsCount.data();

    const T* h   = d.h.data();
    const T* m   = d.m.data();
    const T* x   = d.x.data();
    const T* y   = d.y.data();
    const T* z   = d.z.data();
    const T* rho = d.rho.data();

    T* c11 = d.c11.data();
    T* c12 = d.c12.data();
    T* c13 = d.c13.data();
    T* c22 = d.c22.data();
    T* c23 = d.c23.data();
    T* c33 = d.c33.data();

    const T* wh  = d.wh.data();
    const T* whd = d.whd.data();

    T K         = d.K;
    T sincIndex = d.sincIndex;

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        size_t ni = i - startIndex;
        kernels::IADJLoop3L(i,
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
                            rho,
                            wh,
                            whd,
                            c11,
                            c12,
                            c13,
                            c22,
                            c23,
                            c33);
    }
}

template<class T, class Dataset>
void computeIAD(size_t startIndex, size_t endIndex, size_t ngmax, Dataset& d, const cstone::Box<T>& box)
{
#if defined(USE_CUDA)
    cuda::computeIAD(startIndex, endIndex, ngmax, d, box);
#else
    computeIADImpl(startIndex, endIndex, ngmax, d, box);
#endif
}

} // namespace sph

} // namespace sphexa
