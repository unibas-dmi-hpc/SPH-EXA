#include "sph.cuh"

#include "sph/kernel/density_kern.hpp"

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
 * @param[in]  rho0
 * @param[in]  wrho0
 * @param[out] rho             densities, length @p numParticles
 * @param[out] kx
 * @param[out] whomega
 *
 */
template<class T, class KeyType>
__global__ void cudaDensity(
    T sincIndex,
    T K,
    int ngmax,
    cstone::Box<T> box,
    int firstParticle,
    int lastParticle,
    int numParticles,
    const KeyType* particleKeys,
    const T* x,
    const T* y,
    const T* z,
    const T* h,
    const T* m,
    const T* wh,
    const T* whd,
    const T* rho0,
    const T* wrho0,
    T* rho,
    T* kx,
    T* whomega)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned i   = tid + firstParticle;

    // need to hard-code ngmax stack allocation for now
    assert(ngmax <= NGMAX && "ngmax too big, please increase NGMAX to desired value");
    int neighbors[NGMAX];
    int neighborsCount;

    // starting from CUDA 11.3, dynamic stack allocation is available with the following command
    // int* neighbors = (int*)alloca(ngmax * sizeof(int));

    if (i < lastParticle)
    {
        cstone::findNeighbors(
            i, x, y, z, h, box, cstone::sfcKindPointer(particleKeys), neighbors, &neighborsCount, numParticles, ngmax);

        kernels::densityJLoop(i,
                              sincIndex,
                              K,
                              box,
                              neighbors,
                              neighborsCount,
                              x,
                              y,
                              z,
                              h,
                              m,
                              wh,
                              whd,
                              rho0,
                              wrho0,
                              rho,
                              kx,
                              whomega);
    }
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

    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_x, d.x.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_y, d.y.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_z, d.z.data(), size_np_T, cudaMemcpyHostToDevice));

    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_h, d.h.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_m, d.m.data(), size_np_T, cudaMemcpyHostToDevice));

    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_codes, d.codes.data(), size_np_CodeType, cudaMemcpyHostToDevice));

    for (int i = 0; i < numTasks; ++i)
    {
        int          sIdx   = i % NST;
        cudaStream_t stream = d.devPtrs.d_stream[sIdx].stream;

        unsigned firstParticle       = startIndex + i * taskSize;
        unsigned lastParticle        = std::min(startIndex + (i + 1) * taskSize, endIndex);
        unsigned numParticlesCompute = lastParticle - firstParticle;

        unsigned numThreads = CudaConfig::numThreads;
        unsigned numBlocks  = (numParticlesCompute + numThreads - 1) / numThreads;

        cudaDensity<<<numBlocks, numThreads, 0, stream>>>(d.sincIndex,
                                                          d.K,
                                                          ngmax,
                                                          box,
                                                          firstParticle,
                                                          lastParticle,
                                                          sizeWithHalos,
                                                          d.devPtrs.d_codes,
                                                          d.devPtrs.d_x,
                                                          d.devPtrs.d_y,
                                                          d.devPtrs.d_z,
                                                          d.devPtrs.d_h,
                                                          d.devPtrs.d_m,
                                                          d.devPtrs.d_wh,
                                                          d.devPtrs.d_whd,
                                                          d.devPtrs.d_rho0,
                                                          d.devPtrs.d_wrho0,
                                                          d.devPtrs.d_rho,
                                                          d.devPtrs.d_kx,
                                                          d.devPtrs.d_whomega);
        CHECK_CUDA_ERR(cudaGetLastError());
    }

    // Memcpy in default stream synchronizes all other streams
    CHECK_CUDA_ERR(cudaMemcpy(d.rho.data(),     d.devPtrs.d_rho,     size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.kx.data(),      d.devPtrs.d_kx,      size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.whomega.data(), d.devPtrs.d_whomega, size_np_T, cudaMemcpyDeviceToHost));
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
