#pragma once

#include <vector>

#include <cmath>
#include "math.hpp"
#include "kernels.hpp"
#include "kernel/computeMomentumAndEnergy.hpp"
#ifdef USE_CUDA
#include "cuda/sph.cuh"
#endif

namespace sphexa
{
namespace sph
{

template <typename T, class Dataset>
void computeMomentumAndEnergyIADImpl(const Task& t, Dataset& d, const cstone::Box<T>& box)
{
    size_t numParticles = t.size();
    size_t ngmax = t.ngmax;
    const int* neighbors = t.neighbors.data();
    const int* neighborsCount = t.neighborsCount.data();

    const T* h = d.h.data();
    const T* m = d.m.data();
    const T* x = d.x.data();
    const T* y = d.y.data();
    const T* z = d.z.data();
    const T* vx = d.vx.data();
    const T* vy = d.vy.data();
    const T* vz = d.vz.data();
    const T* ro = d.ro.data();
    const T* c = d.c.data();
    const T* p = d.p.data();

    const T* c11 = d.c11.data();
    const T* c12 = d.c12.data();
    const T* c13 = d.c13.data();
    const T* c22 = d.c22.data();
    const T* c23 = d.c23.data();
    const T* c33 = d.c33.data();

    T* du = d.du.data();
    T* grad_P_x = d.grad_P_x.data();
    T* grad_P_y = d.grad_P_y.data();
    T* grad_P_z = d.grad_P_z.data();
    T* maxvsignal = d.maxvsignal.data();

    const T* wh   = d.wh.data();
    const T* whd  = d.whd.data();
    const T* kx   = d.kx.data();
    const T* rho0 = d.rho0.data();

    const T K = d.K;
    const T sincIndex = d.sincIndex;
    const T Atmin = d.Atmin;
    const T Atmax = d.Atmax;
    const T ramp  = d.ramp;

#if defined(USE_OMP_TARGET)
    // Apparently Cray with -O2 has a bug when calling target regions in a loop. (and computeMomentumAndEnergyIADImpl can be called in a
    // loop). A workaround is to call some method or allocate memory to either prevent buggy optimization or other side effect. with -O1
    // there is no problem Tested with Cray 8.7.3 with NVIDIA Tesla P100 on PizDaint
    std::vector<T> imHereBecauseOfCrayCompilerO2Bug(4, 10);
    const int np = d.x.size();
    const size_t ltsize = d.wh.size();
    const size_t n = numParticles;
    const size_t allNeighbors = n * ngmax;
// clang-format off
#pragma omp target map(to                                                                                                                  \
		       : neighbors[:allNeighbors], neighborsCount[:n], x [0:np], y [0:np], z [0:np],                           \
                       vx [0:np], vy [0:np], vz [0:np], h [0:np], m [0:np], ro [0:np], p [0:np], c [0:np],                                 \
                       c11 [0:np], c12 [0:np], c13 [0:np], c22 [0:np], c23 [0:np], c33 [0:np], wh[0:ltsize], whd[0:ltsize])                                             \
                   map(from                                                                                                                \
                       : grad_P_x [:n], grad_P_y [:n], grad_P_z [:n], du [:n], maxvsignal[0:n])
// clang-format on
#pragma omp teams distribute parallel for
#elif defined(USE_ACC)
    const int np = d.x.size();
    const size_t ltsize = d.wh.size();
    const size_t n = numParticles;
    const size_t allNeighbors = n * ngmax;
// clang-format off
#pragma acc parallel loop copyin(neighbors [0:allNeighbors], neighborsCount [0:n], x [0:np], y [0:np], z [0:np], vx [0:np],   \
                                 vy [0:np], vz [0:np], h [0:np], m [0:np], ro [0:np], p [0:np], c [0:np], c11 [0:np], c12 [0:np],          \
                                 c13 [0:np], c22 [0:np], c23 [0:np], c33 [0:np], wh[0:ltsize], whd[0:ltsize])                                                           \
                          copyout(grad_P_x [:n], grad_P_y [:n], grad_P_z [:n], du [:n], maxvsignal[0:n])
// clang-format on
#else
#pragma omp parallel for schedule(guided)
#endif
    for (size_t pi = 0; pi < numParticles; ++pi)
    {
        int i = pi + t.firstParticle;
        kernels::momentumAndEnergyJLoop(i, sincIndex, K, box, neighbors + ngmax * pi, neighborsCount[pi],
                                        x, y, z, vx, vy, vz, h, m, ro, p, c,
                                        c11, c12, c13, c22, c23, c33, Atmin, Atmax, ramp,
                                        wh, whd, kx, rho0, grad_P_x, grad_P_y, grad_P_z, du, maxvsignal);
    }
}

template <typename T, class Dataset>
void computeMomentumAndEnergyIAD(const std::vector<Task>& taskList, Dataset& d, const cstone::Box<T>& box)
{
#if defined(USE_CUDA)
    cuda::computeMomentumAndEnergyIAD(taskList, d, box);
#else
    for (const auto &task : taskList)
    {
        computeMomentumAndEnergyIADImpl<T>(task, d, box);
    }
#endif
}

} // namespace sph
} // namespace sphexa
