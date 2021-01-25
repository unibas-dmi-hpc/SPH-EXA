#include <cuda.h>
#include <algorithm>

#include "sph.cuh"
#include "BBox.hpp"
#include "ParticlesData.hpp"
#include "cudaUtils.cuh"
#include "../kernels.hpp"
#include "../kernel/computeMomentumAndEnergy.hpp"

namespace sphexa
{
namespace sph
{
namespace cuda
{

template <typename T>
__global__ void computeMomentumAndEnergyIAD(const int n, const T sincIndex, const T K, const int ngmax, const BBox<T> *bbox,
                                            const int *clist, const int *neighbors, const int *neighborsCount, const T *x, const T *y,
                                            const T *z, const T *vx, const T *vy, const T *vz, const T *h, const T *m, const T *ro,
                                            const T *p, const T *c, const T *c11, const T *c12, const T *c13, const T *c22, const T *c23,
                                            const T *c33, const T *wh, const T *whd, const size_t ltsize, T *grad_P_x, T *grad_P_y, T *grad_P_z, T *du, T *maxvsignal)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= n) return;

    sph::kernels::momentumAndEnergyJLoop(tid, sincIndex, K, ngmax, bbox, clist, neighbors, neighborsCount, x, y, z, vx, vy, vz, h, m, ro,
                            p, c, c11, c12, c13, c22, c23, c33, wh, whd, ltsize, grad_P_x, grad_P_y, grad_P_z, du, maxvsignal);
}

template <typename T, class Dataset>
void computeMomentumAndEnergyIAD(const std::vector<Task> &taskList, Dataset &d)
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

    // number of CUDA streams to use
    const int NST = DeviceParticlesData<T, Dataset>::NST;

    const size_t ltsize = d.wh.size();

    /*
    // const size_t size_bbox = sizeof(BBox<T>);
    // const size_t size_np_T = np * sizeof(T);
    // const size_t size_n_int = n * sizeof(int);
    // const size_t size_n_T = n * sizeof(T);
    // const size_t size_allNeighbors = allNeighbors * sizeof(int);

    int *d_clist[NST], *d_neighbors[NST], *d_neighborsCount[NST]; // work arrays per stream

    // input data
    //CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d.d_vx, d.d_vy, d.d_vz, d.d_p, d.d_c, d.d_grad_P_x, d.d_grad_P_y, d.d_grad_P_z, d.d_du, d.d_maxvsignal));
    for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNChunk_int, d_clist[i], d_neighborsCount[i]));
    for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNeighborsChunk_int, d_neighbors[i]));

    cudaStream_t streams[NST];
    for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(cudaStreamCreate(&streams[i]));
    */

    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_vx, d.vx.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_vy, d.vy.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_vz, d.vz.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_p, d.p.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_c, d.c.data(), size_np_T, cudaMemcpyHostToDevice));

    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_c11, d.c11.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_c12, d.c12.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_c13, d.c13.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_c22, d.c22.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_c23, d.c23.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_c33, d.c33.data(), size_np_T, cudaMemcpyHostToDevice));

    //DeviceLinearOctree<T> d_o;
    //d_o.mapLinearOctreeToDevice(o);
    
    for (int i = 0; i < taskList.size(); ++i)
    {
        const auto &t = taskList[i];

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

        computeMomentumAndEnergyIAD<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            n, d.sincIndex, d.K, ngmax, d.devPtrs.d_bbox, d_clist_use, d_neighbors_use, d_neighborsCount_use, d.devPtrs.d_x, d.devPtrs.d_y, d.devPtrs.d_z, d.devPtrs.d_vx, d.devPtrs.d_vy, d.devPtrs.d_vz, d.devPtrs.d_h, d.devPtrs.d_m, d.devPtrs.d_ro,
            d.devPtrs.d_p, d.devPtrs.d_c, d.devPtrs.d_c11, d.devPtrs.d_c12, d.devPtrs.d_c13, d.devPtrs.d_c22, d.devPtrs.d_c23, d.devPtrs.d_c33, d.devPtrs.d_wh, d.devPtrs.d_whd, ltsize, d.devPtrs.d_grad_P_x, d.devPtrs.d_grad_P_y, d.devPtrs.d_grad_P_z, d.devPtrs.d_du, d.devPtrs.d_maxvsignal);

        CHECK_CUDA_ERR(cudaGetLastError());
    }

    //d.devPtrs.d_o.unmapLinearOctreeFromDevice();

    CHECK_CUDA_ERR(cudaMemcpy(d.grad_P_x.data(), d.devPtrs.d_grad_P_x, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.grad_P_y.data(), d.devPtrs.d_grad_P_y, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.grad_P_z.data(), d.devPtrs.d_grad_P_z, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.du.data(), d.devPtrs.d_du, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.maxvsignal.data(), d.devPtrs.d_maxvsignal, size_np_T, cudaMemcpyDeviceToHost));

    /*
   for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(cudaStreamDestroy(streams[i]));

    for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(utils::cudaFree(d_clist[i], d_neighbors[i], d_neighborsCount[i]));
    */
}

template void computeMomentumAndEnergyIAD<double, ParticlesData<double>>(const std::vector<Task> &taskList, ParticlesData<double> &d);

} // namespace cuda
} // namespace sph
} // namespace sphexa
