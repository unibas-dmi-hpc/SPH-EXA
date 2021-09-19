#include "hip/hip_runtime.h"
#include <vector>
#include "Task.hpp"
#include "LinearOctree.hpp"
#include "ParticlesData.hpp"
#include "cudaUtils.cuh"
#include "../kernel/computeFindNeighbors.hpp"

namespace sphexa
{
namespace sph
{
namespace cuda
{

template <typename T>
__global__ void findNeighbors(const DeviceLinearOctree<T> o, const int *clist, const int n, const T *x, const T *y, const T *z, const T *h, const T displx,
                              const T disply, const T displz, const int max, const int may, const int maz, const int ngmax, int *neighbors, int *neighborsCount)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= n) return;

    // Map LinearTree to device pointers
    // Currently OpenMP implementations do not support very well the mapping of structures
    // So we convert everything to simple arrays and pass them to OpenMP
    const int *o_ncells = o.ncells;
    const int *o_cells = o.cells;
    const int *o_localPadding = o.localPadding;
    const int *o_localParticleCount = o.localParticleCount;
    const T *o_xmin = o.xmin;
    const T *o_xmax = o.xmax;
    const T *o_ymin = o.ymin;
    const T *o_ymax = o.ymax;
    const T *o_zmin = o.zmin;
    const T *o_zmax = o.zmax;

    sph::kernels::findNeighborsJLoop(tid, clist, x, y, z, h, displx, disply, displz, max, may, maz, ngmax, neighbors, neighborsCount,
                            // The linear tree
                            o_cells, o_ncells, o_localPadding, o_localParticleCount, o_xmin, o_xmax, o_ymin, o_ymax, o_zmin, o_zmax);
}

template <typename T, class Dataset>
void computeFindNeighbors(const LinearOctree<T> &o, std::vector<Task> &taskList, Dataset &d)
{
    const T *h = d.h.data();
    const T *x = d.x.data();
    const T *y = d.y.data();
    const T *z = d.z.data();

    const int maz = d.bbox.PBCz ? 2 : 0;
    const int may = d.bbox.PBCy ? 2 : 0;
    const int max = d.bbox.PBCx ? 2 : 0;
    
    const T displx = o.xmax[0] - o.xmin[0];
    const T disply = o.ymax[0] - o.ymin[0];
    const T displz = o.zmax[0] - o.zmin[0];

    const size_t np = d.x.size();
    const size_t size_np_T = np * sizeof(T);
    const T ngmax = taskList.empty() ? 0 : taskList.front().ngmax;

    const auto largestChunkSize =
        std::max_element(taskList.cbegin(), taskList.cend(),
                         [](const Task &lhs, const Task &rhs) { return lhs.clist.size() < rhs.clist.size(); })
            ->clist.size();

    const size_t size_largerNeighborsChunk_int = largestChunkSize * ngmax * sizeof(int);
    const size_t size_largerNChunk_int = largestChunkSize * sizeof(int);

    const int NST = 2;

    // Device pointers
    int *d_clist[NST], *d_neighbors[NST], *d_neighborsCount[NST]; // work arrays per stream
    T *d_x, *d_y, *d_z, *d_h;

    CHECK_CUDA_ERR(utils::hipMalloc(size_np_T, d_x, d_y, d_z, d_h));
    for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(utils::hipMalloc(size_largerNChunk_int, d_clist[i], d_neighborsCount[i]));
    for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(utils::hipMalloc(size_largerNeighborsChunk_int, d_neighbors[i]));

    CHECK_CUDA_ERR(hipMemcpy(d_x, d.x.data(), size_np_T, hipMemcpyHostToDevice));
    CHECK_CUDA_ERR(hipMemcpy(d_y, d.y.data(), size_np_T, hipMemcpyHostToDevice));
    CHECK_CUDA_ERR(hipMemcpy(d_z, d.z.data(), size_np_T, hipMemcpyHostToDevice));
    CHECK_CUDA_ERR(hipMemcpy(d_h, d.h.data(), size_np_T, hipMemcpyHostToDevice));

    DeviceLinearOctree<T> d_o;
    d_o.mapLinearOctreeToDevice(o);

    hipStream_t streams[NST];
    for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(hipStreamCreate(&streams[i]));

    for (int i = 0; i < taskList.size(); ++i)
    {
        auto &t = taskList[i];

        const int sIdx = i % NST;
        hipStream_t stream = streams[sIdx];

        int *d_clist_use = d_clist[sIdx];
        int *d_neighbors_use = d_neighbors[sIdx];
        int *d_neighborsCount_use = d_neighborsCount[sIdx];

        const size_t n = t.clist.size();
        const size_t size_n_int = n * sizeof(int);
        const size_t size_nNeighbors = n * ngmax * sizeof(int);

        //CHECK_CUDA_ERR(hipMemcpy(d_clist, t.clist.data(), size_n_int, hipMemcpyHostToDevice));
        CHECK_CUDA_ERR(hipMemcpyAsync(d_clist_use, t.clist.data(), size_n_int, hipMemcpyHostToDevice, stream));

        const int threadsPerBlock = 256;
        const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

        // printf("CUDA Density kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

        hipLaunchKernelGGL(findNeighbors, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, stream, 
            d_o, d_clist_use, n, d_x, d_y, d_z, d_h, displx, disply, displz, max, may, maz, ngmax, d_neighbors_use, d_neighborsCount_use
        );

        CHECK_CUDA_ERR(hipMemcpyAsync(t.neighbors.data(), d_neighbors_use, size_nNeighbors, hipMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERR(hipMemcpyAsync(t.neighborsCount.data(), d_neighborsCount_use, size_n_int, hipMemcpyDeviceToHost, stream));
    }

    for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(hipStreamSynchronize(streams[i]));

    d_o.unmapLinearOctreeFromDevice();

    for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(hipStreamDestroy(streams[i]));

    CHECK_CUDA_ERR(utils::hipFree(d_x, d_y, d_z, d_h));
    for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(utils::hipFree(d_clist[i], d_neighbors[i], d_neighborsCount[i]));
}

template void computeFindNeighbors<double, ParticlesData<double>>(const LinearOctree<double> &o, std::vector<Task> &taskList, ParticlesData<double> &d);
template void computeFindNeighbors<double, ParticlesDataEvrard<double>>(const LinearOctree<double> &o, std::vector<Task> &taskList, ParticlesDataEvrard<double> &d);

} // namespace cuda
} // namespace sph
} // namespace sphexa
