#pragma once

#include "../cuda/cudaParticlesData.cuh"
#include "../lookupTables.hpp"


namespace sphexa
{
namespace sph
{

namespace cuda
{
template <typename T>
__global__ void findNeighbors(const cuda::DeviceLinearOctree<T> o, const int *clist, const int n, const T *x, const T *y, const T *z, const T *h, const T displx,
                                const T disply, const T displz, const int max, const int may, const int maz, const int ngmax, int *neighbors, int *neighborsCount);
}

namespace kernels
{

template <typename T>
CUDA_DEVICE_HOST_FUN inline
T normalize(T d, T min, T max)
{
    return (d - min) / (max - min);
}

template <typename T>
CUDA_DEVICE_HOST_FUN inline
void findNeighborsDispl(const int pi, const int *clist, const T *x, const T *y, const T *z, const T *h,
                        const T displx, const T disply, const T displz, const int ngmax, int *neighbors, int *neighborsCount,
                        // The linear tree
                   		const int *o_cells, const int *o_ncells, const int *o_localPadding, const int *o_localParticleCount, const T *o_xmin, const T *o_xmax, const T *o_ymin, const T *o_ymax, const T *o_zmin, const T *o_zmax)
{
    const int i = clist[pi];

    // // 64 is not enough... Depends on the bucket size and h...
    // // This can be created and stored on the GPU directly.
    // // For a fixed problem and size, if it works then it will always work
    int collisionsCount = 0;
    int collisionNodes[256];

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
        if (o_ncells[node] == 8)
        {
            int mix = std::max((int)(normalize(xi - ri, o_xmin[node], o_xmax[node]) * nX), 0);
            int miy = std::max((int)(normalize(yi - ri, o_ymin[node], o_ymax[node]) * nY), 0);
            int miz = std::max((int)(normalize(zi - ri, o_zmin[node], o_zmax[node]) * nZ), 0);
            int max = std::min((int)(normalize(xi + ri, o_xmin[node], o_xmax[node]) * nX), nX - 1);
            int may = std::min((int)(normalize(yi + ri, o_ymin[node], o_ymax[node]) * nY), nY - 1);
            int maz = std::min((int)(normalize(zi + ri, o_zmin[node], o_zmax[node]) * nZ), nZ - 1);

            // Maximize threads sync
            for (int hz = 0; hz < 2; hz++)
            {
                for (int hy = 0; hy < 2; hy++)
                {
                    for (int hx = 0; hx < 2; hx++)
                    {
                        // if overlap
                        if (hz >= miz && hz <= maz && hy >= miy && hy <= may && hx >= mix && hx <= max)
                        {
                            // int l = hz * nX * nY + hy * nX + hx;
                            // stack[stackptr++] = o_cells[node * 8 + l];
                            const int l = hz * nX * nY + hy * nX + hx;
                            const int child = o_cells[node * 8 + l];
                            if(o_localParticleCount[child] > 0)
                                stack[stackptr++] = child;
                        }
                    }
                }
            }
        }

        if (o_ncells[node] != 8) collisionNodes[collisionsCount++] = node;

        node = stack[--stackptr]; // Pop next
    } while (node > 0);

    //__syncthreads();

    int ngc = neighborsCount[pi];

    for (int ni = 0; ni < collisionsCount; ni++)
    {
        int node = collisionNodes[ni];
        T r2 = ri * ri;

        for (int pj = 0; pj < o_localParticleCount[node]; pj++)
        {
            int j = o_localPadding[node] + pj;

            T xj = x[j];
            T yj = y[j];
            T zj = z[j];

            T xx = xi - xj;
            T yy = yi - yj;
            T zz = zi - zj;

            T dist = xx * xx + yy * yy + zz * zz;

            if (dist < r2 && i != j && ngc < ngmax) neighbors[ngc++] = j;
        }
    }

    neighborsCount[pi] = ngc;

    //__syncthreads();
}

template <typename T>
CUDA_DEVICE_HOST_FUN inline
void findNeighborsJLoop(const int pi, const int *clist, const T *x, const T *y, const T *z, const T *h, const T displx,
                   const T disply, const T displz, const int max, const int may, const int maz, const int ngmax, int *neighbors, int *neighborsCount,
                   // The linear tree
                   const int *o_cells, const int *o_ncells, const int *o_localPadding, const int *o_localParticleCount, const T *o_xmin, const T *o_xmax, const T *o_ymin, const T *o_ymax, const T *o_zmin, const T *o_zmax)
{
    T dispx[3], dispy[3], dispz[3];

    dispx[0] = 0;
    dispy[0] = 0;
    dispz[0] = 0;
    dispx[1] = -displx;
    dispy[1] = -disply;
    dispz[1] = -displz;
    dispx[2] = displx;
    dispy[2] = disply;
    dispz[2] = displz;

    neighborsCount[pi] = 0;

    for (int hz = 0; hz <= maz; hz++)
        for (int hy = 0; hy <= may; hy++)
            for (int hx = 0; hx <= max; hx++)
                findNeighborsDispl<T>(pi, clist, x, y, z, h, dispx[hx], dispy[hy], dispz[hz], ngmax, &neighbors[pi * ngmax], neighborsCount,
                                   // The linear tree
                                   o_cells, o_ncells, o_localPadding, o_localParticleCount, o_xmin, o_xmax, o_ymin, o_ymax, o_zmin, o_zmax);
}

} // namespace kernels
} // namespace sph
} // namespace sphexa
