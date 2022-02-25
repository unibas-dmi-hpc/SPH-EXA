#include <algorithm>

#include "sph.cuh"
#include "particles_data.hpp"
#include "cudaUtils.cuh"
#include "../kernel/density.hpp"

#include "cstone/cuda/findneighbors.cuh"

namespace sphexa
{
namespace sph
{
namespace cuda
{

template<class T, class KeyType>
__global__ void density(T sincIndex, T K, int ngmax, cstone::Box<T> box,
                        int firstParticle, int lastParticle, int numParticles, const KeyType* particleKeys,
                        int* neighborsCount,
                        const T* x, const T* y, const T* z, const T* h, const T* m, const T* wh, const T* whd, T* ro)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned i = tid + firstParticle;
    if (i >= lastParticle) return;

    // need to hard-code ngmax stack allocation for now
    assert(ngmax <= NGMAX && "ngmax too big, please increase NGMAX to desired value");
    int neighbors[NGMAX];
    int neighborsCount_;

    // starting from CUDA 11.3, dynamic stack allocation is available with the following command
    // int* neighbors = (int*)alloca(ngmax * sizeof(int));

    cstone::findNeighbors(
        i, x, y, z, h, box, cstone::sfcKindPointer(particleKeys), neighbors, &neighborsCount_, numParticles, ngmax);

    ro[i] = sph::kernels::densityJLoop(i, sincIndex, K, box, neighbors, neighborsCount_, x, y, z, h, m, wh, whd);

    neighborsCount[tid] = neighborsCount_;
}

template<class Dataset>
void computeDensity(std::vector<Task>& taskList, Dataset& d, const cstone::Box<double>& box)
{
    using T = typename Dataset::RealType;

    size_t numParticles = d.x.size();

    size_t size_np_T = numParticles * sizeof(T);
    size_t size_np_CodeType = numParticles * sizeof(typename Dataset::KeyType);
    T ngmax = taskList.empty() ? 0 : taskList.front().ngmax;

    auto largestChunkSize = std::max_element(taskList.cbegin(),
                                             taskList.cend(),
                                             [](const Task& lhs, const Task& rhs) { return lhs.size() < rhs.size(); })
                                ->size();

    d.devPtrs.resize_streams(largestChunkSize, ngmax);

    // number of CUDA streams to use
    constexpr int NST = DeviceParticlesData<T, Dataset>::NST;

    size_t ltsize = d.wh.size();
    size_t size_lt_T = ltsize * sizeof(T);
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_wh, d.wh.data(), size_lt_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_whd, d.whd.data(), size_lt_T, cudaMemcpyHostToDevice));

    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_x, d.x.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_y, d.y.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_z, d.z.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_h, d.h.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_m, d.m.data(), size_np_T, cudaMemcpyHostToDevice));

    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_codes, d.codes.data(), size_np_CodeType, cudaMemcpyHostToDevice));

    for (int i = 0; i < taskList.size(); ++i)
    {
        auto &t = taskList[i];

        int sIdx = i % NST;
        cudaStream_t stream = d.devPtrs.d_stream[sIdx].stream;

        int* d_neighborsCount_use = d.devPtrs.d_stream[sIdx].d_neighborsCount;

        unsigned firstParticle = t.firstParticle;
        unsigned lastParticle  = t.lastParticle;
        unsigned numParticlesCompute = lastParticle - firstParticle;

        unsigned numThreads = 256;
        unsigned numBlocks  = (numParticlesCompute + numThreads - 1) / numThreads;

        density<<<numBlocks, numThreads, 0, stream>>>(
            d.sincIndex, d.K, t.ngmax, box,
            firstParticle, lastParticle, numParticles, d.devPtrs.d_codes, d_neighborsCount_use,
            d.devPtrs.d_x, d.devPtrs.d_y, d.devPtrs.d_z, d.devPtrs.d_h, d.devPtrs.d_m, d.devPtrs.d_wh, d.devPtrs.d_whd,
            d.devPtrs.d_ro);
        CHECK_CUDA_ERR(cudaGetLastError());

        CHECK_CUDA_ERR(cudaMemcpyAsync(t.neighborsCount.data(), d_neighborsCount_use,
                                       numParticlesCompute * sizeof(int), cudaMemcpyDeviceToHost, stream));
    }

    // Memcpy in default stream synchronizes all other streams
    CHECK_CUDA_ERR(cudaMemcpy(d.ro.data(), d.devPtrs.d_ro, size_np_T, cudaMemcpyDeviceToHost));

}

template void computeDensity(std::vector<Task>&, ParticlesData<double, unsigned>&, const cstone::Box<double>&);
template void computeDensity(std::vector<Task>&, ParticlesData<double, uint64_t>&, const cstone::Box<double>&);

} // namespace cuda
} // namespace sph
} // namespace sphexa

