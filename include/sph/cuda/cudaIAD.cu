#include <algorithm>

#include "sph.cuh"
#include "BBox.hpp"
#include "ParticlesData.hpp"
#include "cudaUtils.cuh"
#include "../kernel/computeIAD.hpp"

#include "cstone/cuda/findneighbors.cuh"

namespace sphexa
{
namespace sph
{
namespace cuda
{
template <typename T>
__global__ void computeIAD(const int n, const T sincIndex, const T K, const int ngmax, const BBox<T> *bbox, const int *clist,
                           const int *neighbors, const int *neighborsCount, const T *x, const T *y, const T *z, const T *h, const T *m, 
                           const T *ro, const T *wh, const T *whd, const size_t ltsize, T *c11, T *c12, T *c13, T *c22, T *c23, T *c33)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= n) return;

    sph::kernels::IADJLoop(tid, sincIndex, K, ngmax, bbox, clist, neighbors, neighborsCount, x, y, z, h, m, ro, wh, whd, ltsize, c11, c12, c13, c22, c23, c33);
}

template <typename T, class Dataset>
void computeIAD(const std::vector<Task> &taskList, Dataset &d)
{
    size_t np = d.x.size();
    size_t size_np_T = np * sizeof(T);
    T ngmax = taskList.empty() ? 0 : taskList.front().ngmax;

    size_t ltsize = d.wh.size();

    auto largestChunkSize =
        std::max_element(taskList.cbegin(), taskList.cend(),
                         [](const Task &lhs, const Task &rhs) { return lhs.clist.size() < rhs.clist.size(); })
            ->clist.size();

    d.devPtrs.resize_streams(largestChunkSize, ngmax);

    // number of CUDA streams to use
    constexpr int NST = DeviceParticlesData<T, Dataset>::NST;

    cstone::Box<T> cstoneBox{d.bbox.xmin, d.bbox.xmax, d.bbox.ymin, d.bbox.ymax, d.bbox.zmin, d.bbox.zmax,
                         d.bbox.PBCx, d.bbox.PBCy, d.bbox.PBCz};

    // CHECK_CUDA_ERR(cudaMemcpy(d_x, d.x.data(), size_np_T, cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERR(cudaMemcpy(d_y, d.y.data(), size_np_T, cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERR(cudaMemcpy(d_z, d.z.data(), size_np_T, cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERR(cudaMemcpy(d_h, d.h.data(), size_np_T, cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERR(cudaMemcpy(d_m, d.m.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_ro, d.ro.data(), size_np_T, cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERR(cudaMemcpy(d_wh, d.wh.data(), size_lt_T, cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERR(cudaMemcpy(d_whd, d.whd.data(), size_lt_T, cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERR(cudaMemcpy(d_bbox, &d.bbox, size_bbox, cudaMemcpyHostToDevice));

    for (int i = 0; i < taskList.size(); ++i)
    {
        const auto &t = taskList[i];

        int sIdx = i % NST;
        cudaStream_t stream = d.devPtrs.d_stream[sIdx].stream;

        int *d_clist_use = d.devPtrs.d_stream[sIdx].d_clist;
        int *d_neighbors_use = d.devPtrs.d_stream[sIdx].d_neighbors;
        int *d_neighborsCount_use = d.devPtrs.d_stream[sIdx].d_neighborsCount;

        size_t n = t.clist.size();
        size_t size_n_int = n * sizeof(int);

        CHECK_CUDA_ERR(cudaMemcpyAsync(d_clist_use, t.clist.data(), size_n_int, cudaMemcpyHostToDevice, stream));

        constexpr int threadsPerBlock = 256;
                  int blocksPerGrid   = (n + threadsPerBlock - 1) / threadsPerBlock;

        findNeighborsCuda(d.devPtrs.d_x, d.devPtrs.d_y, d.devPtrs.d_z, d.devPtrs.d_h, t.clist[0], t.clist[n-1] + 1, np, cstoneBox,
                          d.devPtrs.d_codes, d_neighbors_use, d_neighborsCount_use, ngmax, stream);
        CHECK_CUDA_ERR(cudaGetLastError())

        // printf("CUDA IAD kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

        computeIAD<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(n, d.sincIndex, d.K, ngmax, d.devPtrs.d_bbox, d_clist_use, d_neighbors_use,
            d_neighborsCount_use, d.devPtrs.d_x, d.devPtrs.d_y, d.devPtrs.d_z, d.devPtrs.d_h, d.devPtrs.d_m, d.devPtrs.d_ro, d.devPtrs.d_wh, d.devPtrs.d_whd, ltsize, d.devPtrs.d_c11, d.devPtrs.d_c12, d.devPtrs.d_c13, d.devPtrs.d_c22,
            d.devPtrs.d_c23, d.devPtrs.d_c33);
        CHECK_CUDA_ERR(cudaGetLastError());
    }

    CHECK_CUDA_ERR(cudaMemcpy(d.c11.data(), d.devPtrs.d_c11, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.c12.data(), d.devPtrs.d_c12, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.c13.data(), d.devPtrs.d_c13, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.c22.data(), d.devPtrs.d_c22, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.c23.data(), d.devPtrs.d_c23, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.c33.data(), d.devPtrs.d_c33, size_np_T, cudaMemcpyDeviceToHost));
}

template void computeIAD<double, ParticlesData<double, unsigned>>(const std::vector<Task> &taskList, ParticlesData<double, unsigned> &d);
template void computeIAD<double, ParticlesDataEvrard<double, unsigned>>(const std::vector<Task> &taskList, ParticlesDataEvrard<double, unsigned> &d);
template void computeIAD<double, ParticlesData<double, uint64_t>>(const std::vector<Task> &taskList, ParticlesData<double, uint64_t> &d);
template void computeIAD<double, ParticlesDataEvrard<double, uint64_t>>(const std::vector<Task> &taskList, ParticlesDataEvrard<double, uint64_t> &d);

} // namespace cuda
} // namespace sph
} // namespace sphexa
