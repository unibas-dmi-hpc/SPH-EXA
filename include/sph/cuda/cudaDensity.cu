#include "hip/hip_runtime.h"
#include <algorithm>

#include "sph.cuh"
#include "BBox.hpp"
#include "ParticlesData.hpp"
#include "cudaUtils.cuh"
#include "../kernel/computeDensity.hpp"

#include "cstone/cuda/findneighbors.cuh"

namespace sphexa
{
namespace sph
{
namespace cuda
{

template<class T>
__global__ void density(int n, T sincIndex, T K, int ngmax, const BBox<T>* bbox, const int* clist,
                        const int* neighbors, const int* neighborsCount,
                        const T* x, const T* y, const T* z, const T* h, const T* m, const T* wh, const T* whd,
                        size_t ltsize, T* ro)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= n) return;

    // computes ro[clist[tid]]
    sph::kernels::densityJLoop(tid, sincIndex, K, ngmax, bbox, clist, neighbors, neighborsCount,
                               x, y, z, h, m, wh, whd, ltsize, ro);
}

template<class Dataset>
void computeDensity(std::vector<Task>& taskList, Dataset& d)
{
    using T = typename Dataset::RealType;

    size_t np = d.x.size();
    size_t size_np_T = np * sizeof(T);
    size_t size_np_CodeType = np * sizeof(typename Dataset::KeyType);
    T ngmax = taskList.empty() ? 0 : taskList.front().ngmax;

    auto largestChunkSize =
        std::max_element(taskList.cbegin(), taskList.cend(),
                         [](const Task &lhs, const Task &rhs) { return lhs.clist.size() < rhs.clist.size(); })
            ->clist.size();

    d.devPtrs.resize_streams(largestChunkSize, ngmax);
    size_t size_bbox = sizeof(BBox<T>);
    cstone::Box<T> cstoneBox{d.bbox.xmin, d.bbox.xmax, d.bbox.ymin, d.bbox.ymax, d.bbox.zmin, d.bbox.zmax,
                             d.bbox.PBCx, d.bbox.PBCy, d.bbox.PBCz};

    // number of CUDA streams to use
    constexpr int NST = DeviceParticlesData<T, Dataset>::NST;

    size_t ltsize = d.wh.size();
    size_t size_lt_T = ltsize * sizeof(T);

    CHECK_CUDA_ERR(hipMemcpy(d.devPtrs.d_x, d.x.data(), size_np_T, hipMemcpyHostToDevice));
    CHECK_CUDA_ERR(hipMemcpy(d.devPtrs.d_y, d.y.data(), size_np_T, hipMemcpyHostToDevice));
    CHECK_CUDA_ERR(hipMemcpy(d.devPtrs.d_z, d.z.data(), size_np_T, hipMemcpyHostToDevice));
    CHECK_CUDA_ERR(hipMemcpy(d.devPtrs.d_h, d.h.data(), size_np_T, hipMemcpyHostToDevice));
    CHECK_CUDA_ERR(hipMemcpy(d.devPtrs.d_m, d.m.data(), size_np_T, hipMemcpyHostToDevice));
    CHECK_CUDA_ERR(hipMemcpy(d.devPtrs.d_wh, d.wh.data(), size_lt_T, hipMemcpyHostToDevice));
    CHECK_CUDA_ERR(hipMemcpy(d.devPtrs.d_whd, d.whd.data(), size_lt_T, hipMemcpyHostToDevice));
    CHECK_CUDA_ERR(hipMemcpy(d.devPtrs.d_bbox, &d.bbox, size_bbox, hipMemcpyHostToDevice));

    CHECK_CUDA_ERR(hipMemcpy(d.devPtrs.d_codes, d.codes.data(), size_np_CodeType, hipMemcpyHostToDevice));

    for (int i = 0; i < taskList.size(); ++i)
    {
        auto &t = taskList[i];

        int sIdx = i % NST;
        hipStream_t stream = d.devPtrs.d_stream[sIdx].stream;

        int* d_clist_use = d.devPtrs.d_stream[sIdx].d_clist;
        int* d_neighbors_use = d.devPtrs.d_stream[sIdx].d_neighbors;
        int* d_neighborsCount_use = d.devPtrs.d_stream[sIdx].d_neighborsCount;

        size_t n = t.clist.size();
        size_t size_n_int = n * sizeof(int);

        CHECK_CUDA_ERR(hipMemcpyAsync(d_clist_use, t.clist.data(), size_n_int, hipMemcpyHostToDevice, stream));

        findNeighborsHilbertGpu(d.devPtrs.d_x, d.devPtrs.d_y, d.devPtrs.d_z, d.devPtrs.d_h,
                                t.clist[0], t.clist[n - 1] + 1, np, cstoneBox, d.devPtrs.d_codes,
                                d_neighbors_use, d_neighborsCount_use, ngmax, stream);
        CHECK_CUDA_ERR(hipGetLastError());

        unsigned numThreads = 256;
        unsigned numBlocks  = (n + numThreads - 1) / numThreads;

        // printf("CUDA Density kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

        hipLaunchKernelGGL(density, dim3(numBlocks), dim3(numThreads), 0, stream, 
            n, d.sincIndex, d.K, t.ngmax, d.devPtrs.d_bbox, d_clist_use, d_neighbors_use, d_neighborsCount_use,
            d.devPtrs.d_x, d.devPtrs.d_y, d.devPtrs.d_z, d.devPtrs.d_h, d.devPtrs.d_m, d.devPtrs.d_wh, d.devPtrs.d_whd,
            ltsize, d.devPtrs.d_ro);
        CHECK_CUDA_ERR(hipGetLastError());

        CHECK_CUDA_ERR(hipMemcpyAsync(t.neighborsCount.data(), d_neighborsCount_use,
                                       size_n_int, hipMemcpyDeviceToHost, stream));
    }

    // Memcpy in default stream synchronizes all other streams
    CHECK_CUDA_ERR(hipMemcpy(d.ro.data(), d.devPtrs.d_ro, size_np_T, hipMemcpyDeviceToHost));

}

template void computeDensity(std::vector<Task>&, ParticlesData<double, unsigned>&);
template void computeDensity(std::vector<Task>&, ParticlesDataEvrard<double, unsigned>&);
template void computeDensity(std::vector<Task>&, ParticlesData<double, uint64_t>&);
template void computeDensity(std::vector<Task>&, ParticlesDataEvrard<double, uint64_t>&);

} // namespace cuda
} // namespace sph
} // namespace sphexa
