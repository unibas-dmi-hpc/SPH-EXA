#pragma once

#include <vector>

#include "kernels.hpp"
#include "cuda/sph.cuh"
#include "utils.hpp"

namespace sphexa
{
namespace sph
{

template <typename T, class Dataset>
void computeDensityImpl(const std::vector<int> &l, Dataset &d)
{
    const auto ngmax = d.ngmax;
    const auto K = d.K;
    const auto sincIndex = d.sincIndex;
    const auto bbox = d.bbox;

    const size_t n = l.size();
    const int *clist = l.data();

    const T *h = d.h.data();
    const T *m = d.m.data();
    const T *x = d.x.data();
    const T *y = d.y.data();
    const T *z = d.z.data();

    // I here that indexes to compute in l are sequentially increasing, e.g 0, 1, 2, 3
    const size_t neighborsOffset = l.front() * ngmax;
    const int *neighbors = d.neighbors.data() + neighborsOffset;

    const size_t neighborsCountOffset = l.front();
    const int *neighborsCount = d.neighborsCount.data() + neighborsCountOffset;

    T *ro = d.ro.data();

#if defined(USE_OMP_TARGET)
    const auto allNeighbors = ngmax * n;
    const size_t np = d.x.size();
// clang-format off
#pragma omp target map(to                                                                                                                  \
		       : clist [:n], neighbors [0:allNeighbors], neighborsCount [:n], m [:np], h [:np], x [:np], y [:np], z [:np])         \
                   map(from                                                                                                                \
                       : ro [:n])
// clang-format on
#pragma omp teams distribute parallel for // dist_schedule(guided)
#elif defined(USE_ACC)
    const auto allNeighbors = ngmax * n;
    const size_t np = d.x.size();
#pragma acc parallel loop copyin(n, clist [0:n], neighbors [0:allNeighbors], neighborsCount [0:n], m [0:np], h [0:np], x [0:np], y [0:np], \
                                 z [0:np]) copyout(ro [0:n])
#else
#pragma omp parallel for
#endif
    for (size_t pi = 0; pi < n; ++pi)
    {
        const int i = clist[pi];
        const int nn = neighborsCount[pi];

        T roloc = 0.0;

        // int converstion to avoid a bug that prevents vectorization with some compilers
        for (int pj = 0; pj < nn; pj++)
        {
            const int j = neighbors[pi * ngmax + pj];
            // later can be stores into an array per particle
            const T dist = distancePBC(bbox, h[i], x[i], y[i], z[i], x[j], y[j], z[j]); // store the distance from each neighbor

            // calculate the v as ratio between the distance and the smoothing length
            const T vloc = dist / h[i];

#ifndef NDEBUG
            if (vloc > 2.0 + 1e-6 || vloc < 0.0)
                printf("ERROR:Density(%d,%d) vloc %f -- x %f %f %f -- %f %f %f -- dist %f -- hi %f\n", i, j, vloc, x[i], y[i], z[i], x[j],
                       y[j], z[j], dist, h[i]);
#endif

            const T value = wharmonic(vloc, h[i], sincIndex, K);
            roloc += value * m[j];
        }

        ro[i] = roloc + m[i] * K / (h[i] * h[i] * h[i]);

#ifndef NDEBUG
        if (std::isnan(ro[i])) printf("ERROR::Density(%d) density %f, position: (%f %f %f), h: %f\n", i, ro[i], x[i], y[i], z[i], h[i]);
#endif
    }
}

template <typename T, class Dataset>
void computeDensity(const std::vector<int> &l, Dataset &d)
{
#if defined(USE_CUDA)
    cuda::computeDensity<T>(utils::partition(l, d.noOfGpuLoopSplits), d);
#else
    for (const auto &clist : utils::partition(l, d.noOfGpuLoopSplits))
        computeDensityImpl<T>(clist, d);
#endif
    // for (size_t i = 0; i < d.ro.size(); ++i)
    // {
    //     printf("%d: %.15f", i, d.ro[i]);
    //     if (i % 10 == 0) printf("\n");
    // }
}

template <typename T, class Dataset>
void initFluidDensityAtRest(const std::vector<int> &clist, Dataset &d)
{
    const T *ro = d.ro.data();
    T *ro_0 = d.ro_0.data();

#pragma omp parallel for
    for (size_t pi = 0; pi < clist.size(); ++pi)
    {
        const int i = clist[pi];
        ro_0[i] = ro[i];
    }
}

} // namespace sph
} // namespace sphexa
