#pragma once

#include <vector>

#include "cstone/findneighbors.hpp"

#include "kernels.hpp"
#include "Task.hpp"
#include "kernel/computeDensity.hpp"
#ifdef USE_CUDA
#include "cuda/sph.cuh"
#endif

namespace sphexa
{
namespace sph
{
template <typename T, class Dataset>
void computeDensityImpl(const Task& t, Dataset& d, const cstone::Box<T>& box)
{
    // number of particles in task
    size_t numParticles = t.size();

    const size_t ngmax = t.ngmax;
    const int *neighbors = t.neighbors.data();
    const int *neighborsCount = t.neighborsCount.data();

    const T *h = d.h.data();
    const T *m = d.m.data();
    const T *x = d.x.data();
    const T *y = d.y.data();
    const T *z = d.z.data();

    const T *wh = d.wh.data();
    const T *whd = d.whd.data();

    const T *rho0 = d.rho0.data();
    const T *wrho0 = d.wrho0.data();

    T *kx = d.kx.data();
    T *whomega = d.whomega.data();
    T *ro = d.rho.data();

    const T K = d.K;
    const T sincIndex = d.sincIndex;

#if defined(USE_OMP_TARGET)
    // Apparently Cray with -O2 has a bug when calling target regions in a loop. (and computeDensityImpl can be called in a loop).
    // A workaround is to call some method or allocate memory to either prevent buggy optimization or other side effect.
    // with -O1 there is no problem
    // Tested with Cray 8.7.3 with NVIDIA Tesla P100 on PizDaint
    std::vector<T> imHereBecauseOfCrayCompilerO2Bug(4, 10);

    const size_t np = d.x.size();
    const size_t ltsize = d.wh.size();
    const size_t n = numParticles;
    const size_t allNeighbors = n * ngmax;

// clang-format off
#pragma omp target map(to                                                                                                                  \
                       : n, neighbors [:allNeighbors], neighborsCount [:n], m [0:np], h [0:np], x [0:np], y [0:np], z [0:np],  wh [0:ltsize], whd [0:ltsize])    \
                   map(from                                                                                                                \
                       : ro [:n])
#pragma omp teams distribute parallel for
// clang-format on
#elif defined(USE_ACC)
    const size_t np = d.x.size();
    const size_t ltsize = d.wh.size();
    const size_t n = numParticles;
    const size_t allNeighbors = n * ngmax;
#pragma acc parallel loop copyin(n, neighbors [0:allNeighbors], neighborsCount [0:n], m [0:np], h [0:np], x [0:np], y [0:np], \
                                 z [0:np], wh [0:ltsize], whd [0:ltsize]) copyout(ro[:n])
#else
#pragma omp parallel for
#endif
    for (size_t pi = 0; pi < numParticles; pi++)
    {
        //int neighLoc[ngmax];
        //int count;
        //cstone::findNeighbors(
        //    pi, x, y, z, h, box, cstone::sfcKindPointer(d.codes.data()), neighLoc, &count, d.codes.size(), ngmax);

        int i = pi + t.firstParticle;
        kernels::densityJLoop(i, sincIndex, K, box, neighbors + ngmax * pi, neighborsCount[pi],
                              x, y, z, h, m, wh, whd, rho0, wrho0, ro, kx, whomega);

#ifndef NDEBUG
        if (std::isnan(ro[i]))
            printf("ERROR::Density(%zu) density %f, position: (%f %f %f), h: %f\n", pi, ro[i], x[i], y[i], z[i], h[i]);
#endif
    }
}

template <typename T, class Dataset>
void computeDensity(std::vector<Task> &taskList, Dataset &d, const cstone::Box<T>& box)
{
#if defined(USE_CUDA)
    cuda::computeDensity<Dataset>(taskList, d, box);
#else
    for (const auto &task : taskList)
    {
        computeDensityImpl<T>(task, d, box);
    }
#endif
}

} // namespace sph
} // namespace sphexa
