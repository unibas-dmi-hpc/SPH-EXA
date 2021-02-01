#pragma once

#include <vector>
#include "Task.hpp"
#include "ParticlesData.hpp"
#include "Octree.hpp"
#include "LinearOctree.hpp"
#include "kernel/computeFindNeighbors.hpp"
#include "cuda/sph.cuh"

namespace sphexa
{
namespace sph
{

template <typename T, class Dataset>
void computeFindNeighbors(const LinearOctree<T> &o, std::vector<Task> &taskList, Dataset &d)
{
    const int maz = d.bbox.PBCz ? 2 : 0;
    const int may = d.bbox.PBCy ? 2 : 0;
    const int max = d.bbox.PBCx ? 2 : 0;

    const T displx = o.xmax[0] - o.xmin[0];
    const T disply = o.ymax[0] - o.ymin[0];
    const T displz = o.zmax[0] - o.zmin[0];

    const T ngmax = taskList.empty() ? 0 : taskList.front().ngmax;

    // Device pointers
    const T *d_h = d.h.data();
    const T *d_x = d.x.data();
    const T *d_y = d.y.data();
    const T *d_z = d.z.data();

    // Map LinearTree to device pointers
    // Currently OpenMP implementations do not support very well the mapping of structures
    // So we convert everything to simple arrays and pass them to OpenMP
    const int *o_ncells = o.ncells.data();
    const int *o_cells = o.cells.data();
    const int *o_localPadding = o.localPadding.data();
    const int *o_localParticleCount = o.localParticleCount.data();
    const T *o_xmin = o.xmin.data();
    const T *o_xmax = o.xmax.data();
    const T *o_ymin = o.ymin.data();
    const T *o_ymax = o.ymax.data();
    const T *o_zmin = o.zmin.data();
    const T *o_zmax = o.zmax.data();

    for (auto &t : taskList)
    {
        // Device pointers
        const int *d_clist = t.clist.data();
        int *d_neighbors = t.neighbors.data();
        int *d_neighborsCount = t.neighborsCount.data();

        const size_t n = t.clist.size();
        
// clang-format off
#if defined(USE_OMP_TARGET)
    const size_t np = d.x.size();
    const size_t st = o.size;
    const size_t stt = o.size * 8;
    // Apparently Cray with -O2 has a bug when calling target regions in a loop. (and computeDensityImpl can be called in a loop).
    // A workaround is to call some method or allocate memory to either prevent buggy optimization or other side effect.
    // with -O1 there is no problem
    // Tested with Cray 8.7.3 with NVIDIA Tesla P100 on PizDaint
    std::vector<T> imHereBecauseOfCrayCompilerO2Bug(4, 10);
    
    const size_t nn = n * ngmax;

#pragma omp target map(to : n, ngmax, max, may, maz, displx, disply, displz, d_x[0:np], d_y[0:np], d_z[0:np], d_h[0:np], o_cells[0:stt], o_ncells[0:st], o_localPadding[0:st], o_localParticleCount[0:st], o_xmin[0:st], o_xmax[0:st], o_ymin[0:st], o_ymax[0:st], o_zmin[0:st], o_zmax[0:st], d_clist[0:n]) map(from: d_neighbors[0:nn], d_neighborsCount [0:n])
#pragma omp teams distribute parallel for
// clang-format on
#else
#pragma omp parallel for
#endif
        for (size_t pi = 0; pi < n; pi++)
            kernels::findNeighborsJLoop(pi, d_clist, d_x, d_y, d_z, d_h, displx, disply, displz, max, may, maz, ngmax, d_neighbors, d_neighborsCount,
                                    // The linear tree
                                    o_cells, o_ncells, o_localPadding, o_localParticleCount, o_xmin, o_xmax, o_ymin, o_ymax, o_zmin, o_zmax);
    }
}

template <typename T, class Dataset>
void findNeighbors(const Octree<T> &o, std::vector<Task> &taskList, Dataset &d)
{

#if defined(USE_CUDA)
    //cuda::computeFindNeighbors2<T>(l, taskList, d);
#else
    LinearOctree<T> l;
    createLinearOctree(o, l);
    computeFindNeighbors<T>(l, taskList, d);
#endif
}

size_t neighborsSumImpl(const Task &t)
{
    size_t sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (unsigned int i = 0; i < t.clist.size(); i++)
        sum += t.neighborsCount[i];
    return sum;
}

size_t neighborsSum(const std::vector<Task> &taskList)
{
    size_t sum = 0;
    for (const auto &task : taskList)
    {
        sum += neighborsSumImpl(task);
    }

#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

    return sum;
}
} // namespace sph
} // namespace sphexa
