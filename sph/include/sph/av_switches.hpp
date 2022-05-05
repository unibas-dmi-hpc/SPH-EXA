#pragma once

#include "kernel_ve/av_switches_kern.hpp"
#ifdef USE_CUDA
#include "cuda/sph.cuh"
#endif

namespace sphexa
{
namespace sph
{

template<class T, class Dataset>
void computeAVswitchesImpl(size_t startIndex, size_t endIndex, size_t ngmax, Dataset& d, const cstone::Box<T>& box)
{
    const int* neighbors      = d.neighbors.data();
    const int* neighborsCount = d.neighborsCount.data();

    const T* x  = d.x.data();
    const T* y  = d.y.data();
    const T* z  = d.z.data();
    const T* h  = d.h.data();
    const T* vx = d.vx.data();
    const T* vy = d.vy.data();
    const T* vz = d.vz.data();
    const T* c  = d.c.data();

    const T* c11 = d.c11.data();
    const T* c12 = d.c12.data();
    const T* c13 = d.c13.data();
    const T* c22 = d.c22.data();
    const T* c23 = d.c23.data();
    const T* c33 = d.c33.data();

    const T* divv = d.divv.data();
    const T* wh   = d.wh.data();
    const T* whd  = d.whd.data();
    const T* kx   = d.kx.data();
    const T* xm   = d.xm.data();

    const T K         = d.K;
    const T sincIndex = d.sincIndex;

    const T alphamin       = d.alphamin;
    const T alphamax       = d.alphamax;
    const T decay_constant = d.decay_constant;

    T* alpha = d.alpha.data();

#pragma omp parallel for
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        size_t ni = i - startIndex;
        alpha[i]  = kernels::AVswitchesJLoop(i,
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
                                             c,
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
                                             d.minDt,
                                             alphamin,
                                             alphamax,
                                             decay_constant,
                                             alpha[i]);
    }
}

template<class T, class Dataset>
void computeAVswitches(size_t startIndex, size_t endIndex, size_t ngmax, Dataset& d, const cstone::Box<T>& box)
{
    computeAVswitchesImpl(startIndex, endIndex, ngmax, d, box);
}

} // namespace sph
} // namespace sphexa
