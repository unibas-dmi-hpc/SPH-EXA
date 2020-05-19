#pragma once

#include <vector>
#include "Task.hpp"
#include "ParticlesData.hpp"
#include "Octree.hpp"
#include "LinearOctree.hpp"
#include "cuda/sph.cuh"

namespace sphexa
{
namespace sph
{
template <typename T>
void findNeighborsDispl(const LinearOctree<T> &o, const int *clist, const int pi, const T *x, const T *y, const T *z,  const T *h, const T displx, const T disply, const T displz, const int ngmax,
                      int *neighbors, int *neighborsCount)
{
	const int i = clist[pi];

    // 64 is not enough... Depends on the bucket size and h...
    // This can be created and stored on the GPU directly.
    // For a fixed problem and size, if it works then it will always work
    int collisionsCount = 0;
    int collisionNodes[128];

    const T xi = x[i] + displx;
    const T yi = y[i] + disply;
    const T zi = z[i] + displz;
    const T ri = 2.0 * h[i];

    constexpr int nX = 2;
    constexpr int nY = 2;
    constexpr int nZ = 2;

    int stack[64];
    int stackptr = 0;
    stack[stackptr++] = -1;

    int node = 0;

    do
    {
        if(o.ncells[node] == 8)
        {
            int mix = std::max((int)(normalize(xi - ri, o.xmin[node], o.xmax[node]) * nX), 0);
            int miy = std::max((int)(normalize(yi - ri, o.ymin[node], o.ymax[node]) * nY), 0);
            int miz = std::max((int)(normalize(zi - ri, o.zmin[node], o.zmax[node]) * nZ), 0);
            int max = std::min((int)(normalize(xi + ri, o.xmin[node], o.xmax[node]) * nX), nX - 1);
            int may = std::min((int)(normalize(yi + ri, o.ymin[node], o.ymax[node]) * nY), nY - 1);
            int maz = std::min((int)(normalize(zi + ri, o.zmin[node], o.zmax[node]) * nZ), nZ - 1);

            // Maximize threads sync
            for (int hz = 0; hz <= 2; hz++)
            {
                for (int hy = 0; hy <= 2; hy++)
                {
                    for (int hx = 0; hx <= 2; hx++)
                    {
                        // if overlap
                        if(hz >= miz && hz <= maz && hy >= miy && hy <= may && hx >= mix && hx <= max)
                        {
                            int l = hz * nX * nY + hy * nX + hx;
                            stack[stackptr++] = o.cells[node * 8 +l];
                        }
                    }
                }
            }
        }

        if(o.ncells[node] != 8)
            collisionNodes[collisionsCount++] = node;

        node = stack[--stackptr]; // Pop next
    }
    while(node != -1);

    // SYNCTHREADS

    int ngc = 0;
    
    for(int ni=0; ni<collisionsCount; ni++)
    {
        int node = collisionNodes[ni];
        T r2 = ri * ri;

        for (int pj = 0; pj < o.localParticleCount[node]; pj++)
        {
            int j = o.localPadding[node] + pj;

            T xj = x[j];
            T yj = y[j];
            T zj = z[j];

            T xx = xi - xj;
            T yy = yi - yj;
            T zz = zi - zj;

            T dist = xx * xx + yy * yy + zz * zz;

            if (dist < r2 && i != j && ngc < ngmax)
            	neighbors[ngc++] = j;
        }
    }

    neighborsCount[pi] += ngc;

    // SYNCTHREADS
}

template <typename T>
void findNeighbors(const LinearOctree<T> &o, const int *clist, const int pi, const T *x, const T *y, const T *z, const T *h, const T displx,
                              const T disply, const T displz, const int max, const int may, const int maz, const int ngmax, int *neighbors, int *neighborsCount)
{
	T dispx[3], dispy[3], dispz[3];

    dispx[0] = 0;       dispy[0] = 0;       dispz[0] = 0;
    dispx[1] = -displx; dispy[1] = -disply; dispz[1] = -displz;
    dispx[2] = displx;  dispy[2] = disply;  dispz[2] = displz;

    for (int hz = 0; hz <= maz; hz++)
        for (int hy = 0; hy <= may; hy++)
            for (int hx = 0; hx <= max; hx++)
                findNeighborsDispl(o, clist, pi, x, y, z, h, dispx[hx], dispy[hy], dispz[hz], ngmax, neighbors, neighborsCount);
}

template <typename T, class Dataset>
void findNeighborsImpl(const LinearOctree<T> &o, Task &t, Dataset &d)
{
    const size_t n = t.clist.size();

    const size_t ngmax = t.ngmax;
    const int *clist = t.clist.data();

    const T *h = d.h.data();
    const T *x = d.x.data();
    const T *y = d.y.data();
    const T *z = d.z.data();

    const size_t np = d.x.size();
    const size_t allNeighbors = n * ngmax;

    int *neighbors = t.neighbors.data();
    int *neighborsCount = t.neighborsCount.data();

    const int maz = d.bbox.PBCz? 2 : 0;
    const int may = d.bbox.PBCy? 2 : 0;
    const int max = d.bbox.PBCx? 2 : 0;

    const T displx = o.xmax[0] - o.xmin[0];
    const T disply = o.ymax[0] - o.ymin[0];
    const T displz = o.zmax[0] - o.zmin[0];

//#pragma omp target map(to: x[0:np], y[0:np], z[0:np], h[0:np], neighbors[0:allNeighbors]) map(tofrom: neighborsCount[0:n])
#pragma omp teams distribute parallel for schedule(guided)
    for (size_t pi = 0; pi < n; pi++)
    {
    	neighborsCount[pi] = 0;

        findNeighbors(o, clist, pi, x, y, z, h, displx, disply, displz, max, may, maz, ngmax, &neighbors[pi * ngmax], neighborsCount);

#ifndef NDEBUG
        if (neighborsCount[pi] == 0)
            printf("ERROR::findNodeCollisions(%d) x %f y %f z %f h = %f ngi %d\n", i, x[i], y[i], z[i], h[i], collisionsCount[pi]);
#endif
    }
}

template <typename T, class Dataset>
void findNeighbors(const Octree<T> &o, std::vector<Task> &taskList, Dataset &d)
{
    LinearOctree<T> l;
    createLinearOctree(o, l);

    #if defined(USE_CUDA)
    	cuda::computeFindNeighbors<T>(l, taskList, d);
	#else
	    for (auto &task : taskList)
	    {
	        findNeighborsImpl<T>(l, task, d);
	    }
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
