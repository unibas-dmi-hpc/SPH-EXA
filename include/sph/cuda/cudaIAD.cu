#include <algorithm>

#include "sph.cuh"
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

template<class T, class KeyType>
__global__ void computeIAD(int n, T sincIndex, T K, int ngmax, cstone::Box<T> box, const int* clist,
                           //const int* neighbors, const int* neighborsCount,
                           const KeyType* particleKeys, int numKeys,
                           const T* x, const T* y, const T* z, const T* h, const T* m, const T* ro,
                           const T* wh, const T* whd, T* c11, T* c12, T* c13, T* c22, T* c23, T* c33)
{
    const unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= n) return;

    // need to hard-code ngmax stack allocation for now
    assert(ngmax <= NGMAX && "ngmax too big, please adjust allocation size here to desired value of ngmax");
    int neighbors[NGMAX];

    // starting from CUDA 11.3, dynamic stack allocation is available with the following command
    // int* neighbors = (int*)alloca(ngmax * sizeof(int));

    int neighborsCount;

    int i = clist[tid];

    cstone::findNeighbors(
        i, x, y, z, h, box, cstone::sfcKindPointer(particleKeys), neighbors, &neighborsCount, numKeys, ngmax);

    sph::kernels::IADJLoop(
        i, sincIndex, K, box, neighbors, neighborsCount, x, y, z, h, m, ro, wh, whd, c11, c12, c13, c22, c23, c33);

    // sph::kernels::IADJLoop(i, sincIndex, K, box, neighbors + ngmax * tid, neighborsCount[tid],
    //                        x, y, z, h, m, ro, wh, whd, c11, c12, c13, c22, c23, c33);
}

template <class Dataset>
void computeIAD(const std::vector<Task>& taskList, Dataset& d, const cstone::Box<double>& box)
{
    using T = typename Dataset::RealType;
    size_t np = d.x.size();
    size_t size_np_T = np * sizeof(T);
    T ngmax = taskList.empty() ? 0 : taskList.front().ngmax;

    auto largestChunkSize =
        std::max_element(taskList.cbegin(), taskList.cend(),
                         [](const Task &lhs, const Task &rhs) { return lhs.clist.size() < rhs.clist.size(); })
            ->clist.size();

    d.devPtrs.resize_streams(largestChunkSize, ngmax);

    // number of CUDA streams to use
    constexpr int NST = DeviceParticlesData<T, Dataset>::NST;

    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_ro, d.ro.data(), size_np_T, cudaMemcpyHostToDevice));

    for (int i = 0; i < taskList.size(); ++i)
    {
        const auto &t = taskList[i];

        int sIdx = i % NST;
        cudaStream_t stream = d.devPtrs.d_stream[sIdx].stream;

        int *d_clist_use = d.devPtrs.d_stream[sIdx].d_clist;
        //int *d_neighbors_use = d.devPtrs.d_stream[sIdx].d_neighbors;
        //int *d_neighborsCount_use = d.devPtrs.d_stream[sIdx].d_neighborsCount;

        size_t n = t.clist.size();
        size_t size_n_int = n * sizeof(int);

        CHECK_CUDA_ERR(cudaMemcpyAsync(d_clist_use, t.clist.data(), size_n_int, cudaMemcpyHostToDevice, stream));

        //findNeighborsHilbertGpu(d.devPtrs.d_x, d.devPtrs.d_y, d.devPtrs.d_z, d.devPtrs.d_h, t.clist[0],
        //                        t.clist[n - 1] + 1, np, box, d.devPtrs.d_codes, d_neighbors_use,
        //                        d_neighborsCount_use, ngmax, stream);
        //CHECK_CUDA_ERR(cudaGetLastError());

        unsigned numThreads = 256;
        unsigned numBlocks  = (n + numThreads - 1) / numThreads;

        computeIAD<<<numBlocks, numThreads, 0, stream>>>(
            n, d.sincIndex, d.K, ngmax, box, d_clist_use,
            //d_neighbors_use, d_neighborsCount_use,
            d.devPtrs.d_codes, np,
            d.devPtrs.d_x, d.devPtrs.d_y, d.devPtrs.d_z, d.devPtrs.d_h, d.devPtrs.d_m, d.devPtrs.d_ro,
            d.devPtrs.d_wh, d.devPtrs.d_whd,
            d.devPtrs.d_c11, d.devPtrs.d_c12, d.devPtrs.d_c13, d.devPtrs.d_c22, d.devPtrs.d_c23, d.devPtrs.d_c33);
        CHECK_CUDA_ERR(cudaGetLastError());
    }

    CHECK_CUDA_ERR(cudaMemcpy(d.c11.data(), d.devPtrs.d_c11, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.c12.data(), d.devPtrs.d_c12, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.c13.data(), d.devPtrs.d_c13, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.c22.data(), d.devPtrs.d_c22, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.c23.data(), d.devPtrs.d_c23, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.c33.data(), d.devPtrs.d_c33, size_np_T, cudaMemcpyDeviceToHost));
}

template void computeIAD(const std::vector<Task>& taskList, ParticlesData<double, unsigned>& d,
                         const cstone::Box<double>&);
template void computeIAD(const std::vector<Task>& taskList, ParticlesData<double, uint64_t>& d,
                         const cstone::Box<double>&);

} // namespace cuda
} // namespace sph
} // namespace sphexa
