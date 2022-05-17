#include <algorithm>

#include "sph.cuh"
#include "sph/particles_data.hpp"
#include "cuda_utils.cuh"
#include "sph/kernel/density_kern.hpp"

#include "cstone/cuda/findneighbors.cuh"

namespace sphexa
{
namespace sph
{
namespace cuda
{

template<class T, class KeyType>
__global__ void cudaDensity(T sincIndex, T K, int ngmax, cstone::Box<T> box, int firstParticle, int lastParticle,
                            int numParticles, const KeyType* particleKeys, int* neighborsCount, const T* x, const T* y,
                            const T* z, const T* h, const T* m, const T* wh, const T* whd, T* rho)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned i   = tid + firstParticle;
    if (i >= lastParticle) return;

    // need to hard-code ngmax stack allocation for now
    assert(ngmax <= NGMAX && "ngmax too big, please increase NGMAX to desired value");
    int neighbors[NGMAX];
    int neighborsCount_;

    // starting from CUDA 11.3, dynamic stack allocation is available with the following command
    // int* neighbors = (int*)alloca(ngmax * sizeof(int));

    cstone::findNeighbors(
        i, x, y, z, h, box, cstone::sfcKindPointer(particleKeys), neighbors, &neighborsCount_, numParticles, ngmax);

    rho[i] = sph::kernels::densityJLoop(i, sincIndex, K, box, neighbors, neighborsCount_, x, y, z, h, m, wh, whd);

    neighborsCount[tid] = neighborsCount_;
}

template<class Dataset>
void computeDensity(size_t startIndex, size_t endIndex, size_t ngmax, Dataset& d,
                    const cstone::Box<typename Dataset::RealType>& box)
{
    using T       = typename Dataset::RealType;
    using KeyType = typename Dataset::KeyType;

    size_t sizeWithHalos     = d.x.size();
    size_t numLocalParticles = endIndex - startIndex;
    size_t size_np_T         = sizeWithHalos * sizeof(T);
    size_t size_np_CodeType  = sizeWithHalos * sizeof(KeyType);

    size_t taskSize = DeviceParticlesData<T, KeyType>::taskSize;
    size_t numTasks = iceil(numLocalParticles, taskSize);

    // number of CUDA streams to use
    constexpr int NST = DeviceParticlesData<T, Dataset>::NST;

    CHECK_CUDA_ERR(cudaMemcpy(rawPtr(d.devData.x), d.x.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(rawPtr(d.devData.y), d.y.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(rawPtr(d.devData.z), d.z.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(rawPtr(d.devData.h), d.h.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(rawPtr(d.devData.m), d.m.data(), size_np_T, cudaMemcpyHostToDevice));

    CHECK_CUDA_ERR(cudaMemcpy(rawPtr(d.devData.codes), d.codes.data(), size_np_CodeType, cudaMemcpyHostToDevice));

    for (int i = 0; i < numTasks; ++i)
    {
        int          sIdx   = i % NST;
        cudaStream_t stream = d.devData.d_stream[sIdx].stream;

        int* d_neighborsCount_use = d.devData.d_stream[sIdx].d_neighborsCount;

        unsigned firstParticle       = startIndex + i * taskSize;
        unsigned lastParticle        = std::min(startIndex + (i + 1) * taskSize, endIndex);
        unsigned numParticlesCompute = lastParticle - firstParticle;

        unsigned numThreads = 256;
        unsigned numBlocks  = (numParticlesCompute + numThreads - 1) / numThreads;

        cudaDensity<<<numBlocks, numThreads, 0, stream>>>(d.sincIndex,
                                                          d.K,
                                                          ngmax,
                                                          box,
                                                          firstParticle,
                                                          lastParticle,
                                                          sizeWithHalos,
                                                          rawPtr(d.devData.codes),
                                                          d_neighborsCount_use,
                                                          rawPtr(d.devData.x),
                                                          rawPtr(d.devData.y),
                                                          rawPtr(d.devData.z),
                                                          rawPtr(d.devData.h),
                                                          rawPtr(d.devData.m),
                                                          rawPtr(d.devData.wh),
                                                          rawPtr(d.devData.whd),
                                                          rawPtr(d.devData.rho));
        CHECK_CUDA_ERR(cudaGetLastError());

        CHECK_CUDA_ERR(cudaMemcpyAsync(d.neighborsCount.data() + firstParticle,
                                       d_neighborsCount_use,
                                       numParticlesCompute * sizeof(decltype(d.neighborsCount.front())),
                                       cudaMemcpyDeviceToHost,
                                       stream));
    }

    // Memcpy in default stream synchronizes all other streams
    CHECK_CUDA_ERR(cudaMemcpy(d.rho.data(), rawPtr(d.devData.rho), size_np_T, cudaMemcpyDeviceToHost));
}

template void computeDensity(size_t, size_t, size_t, ParticlesData<double, unsigned, cstone::GpuTag>&,
                             const cstone::Box<double>&);
template void computeDensity(size_t, size_t, size_t, ParticlesData<double, uint64_t, cstone::GpuTag>&,
                             const cstone::Box<double>&);
template void computeDensity(size_t, size_t, size_t, ParticlesData<float, unsigned, cstone::GpuTag>&,
                             const cstone::Box<float>&);
template void computeDensity(size_t, size_t, size_t, ParticlesData<float, uint64_t, cstone::GpuTag>&,
                             const cstone::Box<float>&);

} // namespace cuda
} // namespace sph
} // namespace sphexa
