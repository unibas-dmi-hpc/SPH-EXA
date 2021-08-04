#pragma once

#include <vector>

#include <cmath>
#include "math.hpp"
#include "kernels.hpp"
#include "kernel/computeIAD.hpp"
#ifdef USE_CUDA
#include "cuda/sph.cuh"
#endif

namespace sphexa
{
namespace sph
{

template <typename T, class Dataset>
void computeIADImpl(const Task &t, Dataset &d)
{
    size_t n = t.clist.size();
    size_t ngmax = t.ngmax;
    const int* clist = t.clist.data();
    const int* neighbors = t.neighbors.data();
    const int* neighborsCount = t.neighborsCount.data();

    const T* h = d.h.data();
    const T* m = d.m.data();
    const T* x = d.x.data();
    const T* y = d.y.data();
    const T* z = d.z.data();
    const T* ro = d.ro.data();

    T* c11 = d.c11.data();
    T* c12 = d.c12.data();
    T* c13 = d.c13.data();
    T* c22 = d.c22.data();
    T* c23 = d.c23.data();
    T* c33 = d.c33.data();

    const T* wh = d.wh.data();
    const T* whd = d.whd.data();
    size_t ltsize = d.wh.size();

    const BBox<T>* bbox = &d.bbox;

    T K = d.K;
    T sincIndex = d.sincIndex;

#if defined(USE_OMP_TARGET)
    // Apparently Cray with -O2 has a bug when calling target regions in a loop. (and computeIADImpl can be called in a loop).
    // A workaround is to call some method or allocate memory to either prevent buggy optimization or other side effect.
    // with -O1 there is no problem
    // Tested with Cray 8.7.3 with NVIDIA Tesla P100 on PizDaint
    std::vector<T> imHereBecauseOfCrayCompilerO2Bug(4, 10);
    const int np = d.x.size();
    const size_t allNeighbors = n * ngmax;

// clang-format off
#pragma omp target map(to                                                                                                                  \
		       : clist [:n], neighbors[:allNeighbors], neighborsCount[:n],                                                         \
                       x [0:np], y [0:np], z [0:np], h [0:np], m [0:np], ro [0:np], wh[0:ltsize], whd[0:ltsize])                                                        \
                   map(from                                                                                                                \
                       : c11[:n], c12[:n], c13[:n], c22[:n], c23[:n], c33[:n])
// clang-format on
#pragma omp teams distribute parallel for // dist_schedule(guided)
#elif defined(USE_ACC)
    const int np = d.x.size();
    const size_t allNeighbors = n * ngmax;
// clang-format off
#pragma acc parallel loop copyin(clist [0:n], neighbors [0:allNeighbors], neighborsCount [0:n],                                            \
                                  x [0:np], y [0:np], z [0:np], h [0:np], m [0:np], ro [0:np], wh[0:ltsize], whd[0:ltsize])                                             \
                           copyout(c11 [:n], c12 [:n], c13 [:n], c22 [:n], c23 [:n],                                                       \
                                   c33 [:n])
// clang-format on
#else
#pragma omp parallel for schedule(guided)
#endif
    for (size_t pi = 0; pi < n; ++pi)
    {
        kernels::IADJLoop(pi, sincIndex, K, ngmax, bbox, clist, neighbors, neighborsCount,
                          x, y, z, h, m, ro, wh, whd, ltsize, c11, c12, c13, c22, c23, c33);
    }
}

template <typename T, class Dataset>
void computeIAD(const std::vector<Task> &taskList, Dataset &d)
{
#if defined(USE_CUDA)
    cuda::computeIAD(taskList, d); // utils::partition(l, d.noOfGpuLoopSplits), d);
#else
    for (const auto &task : taskList)
    {
        computeIADImpl<T>(task, d);
    }
#endif
}

} // namespace sph

} // namespace sphexa
