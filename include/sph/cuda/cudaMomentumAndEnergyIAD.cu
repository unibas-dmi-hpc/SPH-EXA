#include <algorithm>

#include "sph.cuh"
#include "ParticlesData.hpp"
#include "cudaUtils.cuh"
#include "../kernel/computeMomentumAndEnergy.hpp"

#include "cstone/cuda/findneighbors.cuh"

namespace sphexa
{
namespace sph
{
namespace cuda
{

template<class T, class KeyType>
__global__ void computeMomentumAndEnergyIAD(T sincIndex, T K, int ngmax, cstone::Box<T> box,
                                            //const int* neighbors, const int* neighborsCount,
                                            int firstParticle, int lastParticle, int numParticles,
                                            const KeyType* particleKeys,
                                            const T* x, const T* y, const T* z, const T* vx, const T* vy, const T* vz,
                                            const T* h, const T* m, const T* ro, const T* p, const T* c,
                                            const T* c11, const T* c12, const T* c13, const T* c22, const T* c23,
                                            const T* c33, const T* wh, const T* whd,
                                            T* grad_P_x, T* grad_P_y, T* grad_P_z, T* du, T* maxvsignal)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned i = tid + firstParticle;
    if (i >= lastParticle) return;

    // need to hard-code ngmax stack allocation for now
    assert(ngmax <= NGMAX && "ngmax too big, please increase NGMAX to desired value");
    int neighbors[NGMAX];
    int neighborsCount;

    // starting from CUDA 11.3, dynamic stack allocation is available with the following command
    // int* neighbors = (int*)alloca(ngmax * sizeof(int));

    cstone::findNeighbors(
        i, x, y, z, h, box, cstone::sfcKindPointer(particleKeys), neighbors, &neighborsCount, numParticles, ngmax);

    sph::kernels::momentumAndEnergyJLoop(i, sincIndex, K, box, neighbors, neighborsCount,
                                         x, y, z, vx, vy, vz, h, m, ro, p, c, c11, c12, c13, c22, c23, c33,
                                         wh, whd, grad_P_x, grad_P_y, grad_P_z, du, maxvsignal);
}

template <class Dataset>
void computeMomentumAndEnergyIAD(const std::vector<Task> &taskList, Dataset &d, const cstone::Box<double>& box)
{
    using T = typename Dataset::RealType;

    size_t numParticles = d.x.size();
    size_t size_np_T = numParticles * sizeof(T);
    T ngmax = taskList.empty() ? 0 : taskList.front().ngmax;

    auto largestChunkSize = std::max_element(taskList.cbegin(),
                                             taskList.cend(),
                                             [](const Task& lhs, const Task& rhs) { return lhs.size() < rhs.size(); })
                                ->size();

    d.devPtrs.resize_streams(largestChunkSize, ngmax);

    // number of CUDA streams to use
    constexpr int NST = DeviceParticlesData<T, Dataset>::NST;

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

    for (int i = 0; i < taskList.size(); ++i)
    {
        const auto& t = taskList[i];

        int sIdx = i % NST;
        cudaStream_t stream = d.devPtrs.d_stream[sIdx].stream;

        //int* d_neighborsCount_use = d.devPtrs.d_stream[sIdx].d_neighborsCount;

        unsigned firstParticle = t.firstParticle;
        unsigned lastParticle  = t.lastParticle;
        unsigned numParticlesCompute = lastParticle - firstParticle;

        unsigned numThreads = 128;
        unsigned numBlocks  = (numParticlesCompute + numThreads - 1) / numThreads;

        computeMomentumAndEnergyIAD<<<numBlocks, numThreads, 0, stream>>>(
            d.sincIndex, d.K, ngmax, box, //d_neighborsCount_use,
            firstParticle, lastParticle, numParticles, d.devPtrs.d_codes,
            d.devPtrs.d_x, d.devPtrs.d_y, d.devPtrs.d_z, d.devPtrs.d_vx, d.devPtrs.d_vy, d.devPtrs.d_vz,
            d.devPtrs.d_h, d.devPtrs.d_m, d.devPtrs.d_ro, d.devPtrs.d_p, d.devPtrs.d_c,
            d.devPtrs.d_c11, d.devPtrs.d_c12, d.devPtrs.d_c13, d.devPtrs.d_c22, d.devPtrs.d_c23, d.devPtrs.d_c33,
            d.devPtrs.d_wh, d.devPtrs.d_whd, d.devPtrs.d_grad_P_x, d.devPtrs.d_grad_P_y, d.devPtrs.d_grad_P_z,
            d.devPtrs.d_du, d.devPtrs.d_maxvsignal);

        CHECK_CUDA_ERR(cudaGetLastError());
    }

    CHECK_CUDA_ERR(cudaMemcpy(d.grad_P_x.data(), d.devPtrs.d_grad_P_x, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.grad_P_y.data(), d.devPtrs.d_grad_P_y, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.grad_P_z.data(), d.devPtrs.d_grad_P_z, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.du.data(), d.devPtrs.d_du, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.maxvsignal.data(), d.devPtrs.d_maxvsignal, size_np_T, cudaMemcpyDeviceToHost));
}

template void computeMomentumAndEnergyIAD(const std::vector<Task>& taskList, ParticlesData<double, unsigned>& d,
                                          const cstone::Box<double>&);
template void computeMomentumAndEnergyIAD(const std::vector<Task>& taskList, ParticlesData<double, uint64_t>& d,
                                          const cstone::Box<double>&);

} // namespace cuda
} // namespace sph
} // namespace sphexa
