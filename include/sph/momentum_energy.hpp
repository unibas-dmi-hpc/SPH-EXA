#pragma once

#include <vector>

#include <cmath>
#include "math.hpp"
#include "kernels.hpp"
#include "kernel/grad_p.hpp"
#ifdef USE_CUDA
#include "cuda/sph.cuh"
#endif

namespace sphexa
{
namespace sph
{

template<class T, class Dataset>
void computeMomentumAndEnergyImpl(size_t startIndex, size_t endIndex, size_t ngmax, Dataset& d,
                                  const cstone::Box<T>& box)
{
    const int* neighbors      = d.neighbors.data();
    const int* neighborsCount = d.neighborsCount.data();

    const T* h   = d.h.data();
    const T* m   = d.m.data();
    const T* x   = d.x.data();
    const T* y   = d.y.data();
    const T* z   = d.z.data();
    const T* vx  = d.vx.data();
    const T* vy  = d.vy.data();
    const T* vz  = d.vz.data();
    const T* rho = d.rho.data();
    const T* c   = d.c.data();
    const T* p   = d.p.data();

    const T* c11 = d.c11.data();
    const T* c12 = d.c12.data();
    const T* c13 = d.c13.data();
    const T* c22 = d.c22.data();
    const T* c23 = d.c23.data();
    const T* c33 = d.c33.data();

    T* du         = d.du.data();
    T* grad_P_x   = d.grad_P_x.data();
    T* grad_P_y   = d.grad_P_y.data();
    T* grad_P_z   = d.grad_P_z.data();
    T* maxvsignal = d.maxvsignal.data();

    const T* wh  = d.wh.data();
    const T* whd = d.whd.data();

    T K         = d.K;
    T sincIndex = d.sincIndex;

#pragma omp parallel for schedule(static)
    for (size_t i = startIndex; i < endIndex; ++i)
    {
        size_t ni = i - startIndex;
        kernels::momentumAndEnergyJLoop(i,
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
                                        m,
                                        rho,
                                        p,
                                        c,
                                        c11,
                                        c12,
                                        c13,
                                        c22,
                                        c23,
                                        c33,
                                        wh,
                                        whd,
                                        grad_P_x,
                                        grad_P_y,
                                        grad_P_z,
                                        du,
                                        maxvsignal);
    }
}

template<class T, class Dataset>
void computeMomentumAndEnergy(size_t startIndex, size_t endIndex, size_t ngmax, Dataset& d, const cstone::Box<T>& box)
{
#if defined(USE_CUDA)
    cuda::computeMomentumAndEnergy(startIndex, endIndex, ngmax, d, box);
#else
    computeMomentumAndEnergyImpl(startIndex, endIndex, ngmax, d, box);
#endif
}

} // namespace sph
} // namespace sphexa
