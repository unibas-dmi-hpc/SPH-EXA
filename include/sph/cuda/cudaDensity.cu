#include <algorithm>

#include "sph.cuh"
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

template<class T/*, class KeyType*/>
__global__ void density(int n, T sincIndex, T K, int ngmax, cstone::Box<T> box, const int* clist,
                        const int* neighbors, const int* neighborsCount,
                        //const KeyType* particleKeys, int numKeys,
                        const T* x, const T* y, const T* z, const T* h, const T* m, const T* wh, const T* whd, T* ro)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= n) return;

    //int nLoc[150];
    //int nCount;

    int i = clist[tid];

    //cstone::findNeighbors(i, x, y ,z, h, box, cstone::sfcKindPointer(particleKeys), nLoc, &nCount, numKeys, ngmax);

    //ro[i] = sph::kernels::densityJLoop(i, sincIndex, K, box, nLoc, nCount, x, y, z, h, m, wh, whd);
    ro[i] = sph::kernels::densityJLoop(
        i, sincIndex, K, box, neighbors + ngmax * tid, neighborsCount[tid], x, y, z, h, m, wh, whd);
}

template<class Dataset>
void computeDensity(std::vector<Task>& taskList, Dataset& d, const cstone::Box<double>& box)
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

    // number of CUDA streams to use
    constexpr int NST = DeviceParticlesData<T, Dataset>::NST;

    size_t ltsize = d.wh.size();
    size_t size_lt_T = ltsize * sizeof(T);

    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_x, d.x.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_y, d.y.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_z, d.z.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_h, d.h.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_m, d.m.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_wh, d.wh.data(), size_lt_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_whd, d.whd.data(), size_lt_T, cudaMemcpyHostToDevice));

    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_codes, d.codes.data(), size_np_CodeType, cudaMemcpyHostToDevice));

    for (int i = 0; i < taskList.size(); ++i)
    {
        auto &t = taskList[i];

        int sIdx = i % NST;
        cudaStream_t stream = d.devPtrs.d_stream[sIdx].stream;

        int* d_clist_use = d.devPtrs.d_stream[sIdx].d_clist;
        int* d_neighbors_use = d.devPtrs.d_stream[sIdx].d_neighbors;
        int* d_neighborsCount_use = d.devPtrs.d_stream[sIdx].d_neighborsCount;

        size_t n = t.clist.size();
        size_t size_n_int = n * sizeof(int);

        CHECK_CUDA_ERR(cudaMemcpyAsync(d_clist_use, t.clist.data(), size_n_int, cudaMemcpyHostToDevice, stream));

        findNeighborsHilbertGpu(d.devPtrs.d_x, d.devPtrs.d_y, d.devPtrs.d_z, d.devPtrs.d_h,
                                t.clist[0], t.clist[n - 1] + 1, np, box, d.devPtrs.d_codes,
                                d_neighbors_use, d_neighborsCount_use, ngmax, stream);
        CHECK_CUDA_ERR(cudaGetLastError());

        unsigned numThreads = 256;
        unsigned numBlocks  = (n + numThreads - 1) / numThreads;

        density<<<numBlocks, numThreads, 0, stream>>>(
            n, d.sincIndex, d.K, t.ngmax, box, d_clist_use, d_neighbors_use, d_neighborsCount_use,
            //d.devPtrs.d_codes, np,
            d.devPtrs.d_x, d.devPtrs.d_y, d.devPtrs.d_z, d.devPtrs.d_h, d.devPtrs.d_m, d.devPtrs.d_wh, d.devPtrs.d_whd,
            d.devPtrs.d_ro);
        CHECK_CUDA_ERR(cudaGetLastError());

        CHECK_CUDA_ERR(cudaMemcpyAsync(t.neighborsCount.data(), d_neighborsCount_use,
                                       size_n_int, cudaMemcpyDeviceToHost, stream));
    }

    // Memcpy in default stream synchronizes all other streams
    CHECK_CUDA_ERR(cudaMemcpy(d.ro.data(), d.devPtrs.d_ro, size_np_T, cudaMemcpyDeviceToHost));

}

template void computeDensity(std::vector<Task>&, ParticlesData<double, unsigned>&, const cstone::Box<double>&);
template void computeDensity(std::vector<Task>&, ParticlesData<double, uint64_t>&, const cstone::Box<double>&);

} // namespace cuda
} // namespace sph
} // namespace sphexa
