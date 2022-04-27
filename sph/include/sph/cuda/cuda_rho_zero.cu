#include "sph.cuh"

#include "sph/kernel/rho_zero_kern.hpp"

namespace sphexa
{
namespace sph
{
namespace cuda
{

/*! @brief
 *
 * @tparam     T               float or double
 * @tparam     KeyType         32- or 64-bit unsigned integer
 * @param[in]  sincIndex
 * @param[in]  K
 * @param[in]  ngmax           maximum number of neighbors per particle to use
 * @param[in]  box             global coordinate bounding box
 * @param[in]  firstParticle   first particle to compute
 * @param[in]  lastParticle    last particle to compute
 * @param[in]  numParticles    number of local particles + halos
 * @param[in]  particleKeys    SFC keys of particles, sorted in ascending order
 * @param[in]  x               x coords, length @p numParticles, SFC sorted
 * @param[in]  y               y coords, length @p numParticles, SFC sorted
 * @param[in]  z               z coords, length @p numParticles, SFC sorted
 * @param[in]  h               smoothing lengths, length @p numParticles
 * @param[in]  m               masses, length @p numParticles
 * @param[in]  wh              sinc lookup table
 * @param[in]  whd             sinc derivative lookup table
 * @param[out] rho0
 * @param[out] wrho0
 *
 */
template<class T, class KeyType>
__global__ void cudaRhoZero(
    T sincIndex,
    T K,
    int ngmax,
    cstone::Box<T> box,
    int firstParticle,
    int lastParticle,
    int numParticles,
    const KeyType* particleKeys,
    int* neighborsCount,
    const T* x,
    const T* y,
    const T* z,
    const T* h,
    const T* m,
    const T* wh,
    const T* whd,
    T* rho0,
    T* wrho0)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned i   = tid + firstParticle;

    // need to hard-code ngmax stack allocation for now
    assert(ngmax <= CudaConfig::NGMAX && "ngmax too big, please increase NGMAX to desired value");
    int neighbors[CudaConfig::NGMAX];
    int neighborsCount_;

    // starting from CUDA 11.3, dynamic stack allocation is available with the following command
    // int* neighbors = (int*)alloca(ngmax * sizeof(int));

    if (i < lastParticle)
    {
        cstone::findNeighbors(
            i, x, y, z, h, box, cstone::sfcKindPointer(particleKeys), neighbors, &neighborsCount_, numParticles, ngmax);

        kernels::rhoZeroJLoop(i,
                              sincIndex,
                              K,
                              box,
                              neighbors,
                              neighborsCount_,
                              x,
                              y,
                              z,
                              h,
                              m,
                              wh,
                              whd,
                              rho0,
                              wrho0);

        neighborsCount[tid] = neighborsCount_;
    }
}

template<class Dataset>
void computeRhoZero(size_t startIndex, size_t endIndex, size_t ngmax, Dataset& d,
                    const cstone::Box<typename Dataset::RealType>& box)
{
    using T       = typename Dataset::RealType;
    using KeyType = typename Dataset::KeyType;

    size_t sizeWithHalos     = d.x.size();
    size_t numLocalParticles = endIndex - startIndex;

    size_t taskSize = DeviceParticlesData<T, KeyType>::taskSize;
    size_t numTasks = iceil(numLocalParticles, taskSize);

    // number of CUDA streams to use
    constexpr int NST = DeviceParticlesData<T, Dataset>::NST;

    for (int i = 0; i < numTasks; ++i)
    {
        int          sIdx   = i % NST;
        cudaStream_t stream = d.devPtrs.d_stream[sIdx].stream;

        unsigned firstParticle       = startIndex + i * taskSize;
        unsigned lastParticle        = std::min(startIndex + (i + 1) * taskSize, endIndex);
        unsigned numParticlesCompute = lastParticle - firstParticle;

        unsigned numThreads = CudaConfig::numThreads;
        unsigned numBlocks  = (numParticlesCompute + numThreads - 1) / numThreads;

        int* d_neighborsCount_use = d.devPtrs.d_stream[sIdx].d_neighborsCount;

        cudaRhoZero<<<numBlocks, numThreads, 0, stream>>>(d.sincIndex,
                                                          d.K,
                                                          ngmax,
                                                          box,
                                                          firstParticle,
                                                          lastParticle,
                                                          sizeWithHalos,
                                                          d.devPtrs.d_codes,
                                                          d_neighborsCount_use,
                                                          d.devPtrs.d_x,
                                                          d.devPtrs.d_y,
                                                          d.devPtrs.d_z,
                                                          d.devPtrs.d_h,
                                                          d.devPtrs.d_m,
                                                          d.devPtrs.d_wh,
                                                          d.devPtrs.d_whd,
                                                          d.devPtrs.d_rho0,
                                                          d.devPtrs.d_wrho0);
        CHECK_CUDA_ERR(cudaGetLastError());

        CHECK_CUDA_ERR(cudaMemcpyAsync(d.neighborsCount.data() + firstParticle,
                                       d_neighborsCount_use,
                                       numParticlesCompute * sizeof(decltype(d.neighborsCount.front())),
                                       cudaMemcpyDeviceToHost,
                                       stream));
    }
}

template void computeRhoZero(size_t, size_t, size_t, ParticlesData<double, unsigned, cstone::GpuTag>&,
                             const cstone::Box<double>&);
template void computeRhoZero(size_t, size_t, size_t, ParticlesData<double, uint64_t, cstone::GpuTag>&,
                             const cstone::Box<double>&);
template void computeRhoZero(size_t, size_t, size_t, ParticlesData<float, unsigned, cstone::GpuTag>&,
                             const cstone::Box<float>&);
template void computeRhoZero(size_t, size_t, size_t, ParticlesData<float, uint64_t, cstone::GpuTag>&,
                             const cstone::Box<float>&);

} // namespace cuda
} // namespace sph
} // namespace sphexa
