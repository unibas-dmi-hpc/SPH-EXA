#include <cuda.h>
#include <algorithm>

#include "sph.cuh"
#include "BBox.hpp"
#include "ParticlesData.hpp"
#include "cudaUtils.cuh"
#include "../kernels.hpp"
#include "../kernel/computeDensity.hpp"

namespace sphexa
{
namespace sph
{
namespace cuda
{
template <typename T>
__global__ void density(const int n, const T sincIndex, const T K, const int ngmax, const BBox<T> *bbox, const int *clist,
                        const int *neighbors, const int *neighborsCount, const T *x, const T *y, const T *z, const T *h, const T *m, const T *wh, const T *whd, const size_t ltsize, T *ro)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= n) return;

    // computes ro[i]
    sph::kernels::densityJLoop(tid, sincIndex, K, ngmax, bbox, clist, neighbors, neighborsCount, x, y, z, h, m, wh, whd, ltsize, ro);
}

template <typename T, class Dataset>
void computeDensity(std::vector<Task> &taskList, Dataset &d)
{
    const int maz = d.bbox.PBCz ? 2 : 0;
    const int may = d.bbox.PBCy ? 2 : 0;
    const int max = d.bbox.PBCx ? 2 : 0;
    
    const T displx = d.devPtrs.d_o.xmax0 - d.devPtrs.d_o.xmin0;
    const T disply = d.devPtrs.d_o.ymax0 - d.devPtrs.d_o.ymin0;
    const T displz = d.devPtrs.d_o.zmax0 - d.devPtrs.d_o.zmin0;

    const size_t np = d.x.size();
    const size_t size_np_T = np * sizeof(T);
    const T ngmax = taskList.empty() ? 0 : taskList.front().ngmax;

    const auto largestChunkSize =
        std::max_element(taskList.cbegin(), taskList.cend(),
                         [](const Task &lhs, const Task &rhs) { return lhs.clist.size() < rhs.clist.size(); })
            ->clist.size();

    d.devPtrs.resize_streams(largestChunkSize, ngmax);
    /*
    const size_t size_largerNeighborsChunk_int = largestChunkSize * ngmax * sizeof(int);
    const size_t size_largerNChunk_int = largestChunkSize * sizeof(int);
    */
    const size_t size_bbox = sizeof(BBox<T>);

    // number of CUDA streams to use
    const int NST = DeviceParticlesData<T, Dataset>::NST;

    const size_t ltsize = d.wh.size();
    const size_t size_lt_T = ltsize * sizeof(T);

    /*
    // initialize streams
    cudaStream_t streams[NST];
    for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(cudaStreamCreate(&streams[i]));

    // device pointers - d_ prefix stands for device
    int *d_clist[NST], *d_neighbors[NST], *d_neighborsCount[NST];

    // input data
    CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d.d_x, d.d_y, d.d_z, d.d_h, d.d_m, d.d_ro));
    CHECK_CUDA_ERR(utils::cudaMalloc(size_lt_T, d.d_wh, d.d_whd));
    CHECK_CUDA_ERR(utils::cudaMalloc(size_bbox, d.d_bbox));

    for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNChunk_int, d_clist[i], d_neighborsCount[i]));
    for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNeighborsChunk_int, d_neighbors[i]));
    //CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNChunk_int, d_clist, d_neighborsCount));
    //CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNeighborsChunk_int, d_neighbors));
    */

    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_x, d.x.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_y, d.y.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_z, d.z.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_h, d.h.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_m, d.m.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_wh, d.wh.data(), size_lt_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_whd, d.whd.data(), size_lt_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_bbox, &d.bbox, size_bbox, cudaMemcpyHostToDevice));

    //DeviceLinearOctree<T> d_o;
    //d.devPtrs.d_o.mapLinearOctreeToDevice(o);

    for (int i = 0; i < taskList.size(); ++i)
    {
        auto &t = taskList[i];

        const int sIdx = i % NST;
        /*
        cudaStream_t stream = streams[sIdx];

        int *d_clist_use = d_clist[sIdx];
        int *d_neighbors_use = d_neighbors[sIdx];
        int *d_neighborsCount_use = d_neighborsCount[sIdx];
        */
        cudaStream_t stream = d.devPtrs.d_stream[sIdx].stream;

        int *d_clist_use = d.devPtrs.d_stream[sIdx].d_clist;
        int *d_neighbors_use = d.devPtrs.d_stream[sIdx].d_neighbors;
        int *d_neighborsCount_use = d.devPtrs.d_stream[sIdx].d_neighborsCount;

        const size_t n = t.clist.size();
        const size_t size_n_int = n * sizeof(int);
        // const size_t size_nNeighbors = n * ngmax * sizeof(int);

        CHECK_CUDA_ERR(cudaMemcpyAsync(d_clist_use, t.clist.data(), size_n_int, cudaMemcpyHostToDevice, stream));
        //CHECK_CUDA_ERR(cudaMemcpyAsync(d_neighbors_use, t.neighbors.data(), size_nNeighbors, cudaMemcpyHostToDevice, stream));
        //CHECK_CUDA_ERR(cudaMemcpyAsync(d_neighborsCount_use, t.neighborsCount.data(), size_n_int, cudaMemcpyHostToDevice, stream));

        const int threadsPerBlock = 256;
        const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

        kernels::findNeighbors<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            d.devPtrs.d_o, d_clist_use, n, d.devPtrs.d_x, d.devPtrs.d_y, d.devPtrs.d_z, d.devPtrs.d_h, displx, disply, displz, max, may, maz, ngmax, d_neighbors_use, d_neighborsCount_use
        );

        // printf("CUDA Density kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

        density<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(n, d.sincIndex, d.K, t.ngmax, d.devPtrs.d_bbox, d_clist_use, d_neighbors_use, d_neighborsCount_use,
            d.devPtrs.d_x, d.devPtrs.d_y, d.devPtrs.d_z, d.devPtrs.d_h, d.devPtrs.d_m, d.devPtrs.d_wh, d.devPtrs.d_whd, ltsize, d.devPtrs.d_ro);
        CHECK_CUDA_ERR(cudaGetLastError());

        CHECK_CUDA_ERR(cudaMemcpyAsync(t.neighborsCount.data(), d_neighborsCount_use, size_n_int, cudaMemcpyDeviceToHost, stream));
    }

    //d_o.unmapLinearOctreeFromDevice();

    // Memcpy in default stream synchronizes all other streams
    CHECK_CUDA_ERR(cudaMemcpy(d.ro.data(), d.devPtrs.d_ro, size_np_T, cudaMemcpyDeviceToHost));

    /*
    for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(cudaStreamDestroy(streams[i]));

    for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(utils::cudaFree(d_clist[i], d_neighbors[i], d_neighborsCount[i]));
    */
}

template void computeDensity<double, ParticlesData<double>>(std::vector<Task> &taskList, ParticlesData<double> &d);

} // namespace cuda
} // namespace sph
} // namespace sphexa
