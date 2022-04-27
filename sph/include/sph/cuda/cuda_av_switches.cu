#include "sph.cuh"

#include "sph/kernel/av_switches_kern.hpp"

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
 * @param[in]  vx
 * @param[in]  vy
 * @param[in]  vz
 * @param[in]  h               smoothing lengths, length @p numParticles
 * @param[in]  m               masses, length @p numParticles
 * @param[in]  c
 * @param[in]  c11             IAD components, length @p numParticles
 * @param[in]  c12
 * @param[in]  c13
 * @param[in]  c22
 * @param[in]  c23
 * @param[in]  c33
 * @param[in]  wh              sinc lookup table
 * @param[in]  whd             sinc derivative lookup table
 * @param[in]  kx
 * @param[in]  rho0
 * @param[in]  divv
 * @param[in]  dt
 * @param[in]  alphamin
 * @param[in]  alphamax
 * @param[in]  decay_constant
 * @param[out] alpha_i
 *
 */
template<class T, class KeyType>
__global__ void cudaAVswitches(
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
        const T* vx,
        const T* vy,
        const T* vz,
        const T* h,
        const T* m,
        const T* c,
        const T* c11,
        const T* c12,
        const T* c13,
        const T* c22,
        const T* c23,
        const T* c33,
        const T* wh,
        const T* whd,
        const T* kx,
        const T* rho0,
        const T* divv,
        const T dt,
        const T alphamin,
        const T alphamax,
        const T decay_constant,
        T* alpha)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned i   = tid + firstParticle;

    // need to hard-code ngmax stack allocation for now
    assert(ngmax <= CudaConfig::NGMAX && "ngmax too big, please increase NGMAX to desired value");
    int neighbors[CudaConfig::NGMAX];
    int neighborsCount;

    // starting from CUDA 11.3, dynamic stack allocation is available with the following command
    // int* neighbors = (int*)alloca(ngmax * sizeof(int));

    if (i < lastParticle)
    {
        cstone::findNeighbors(
            i, x, y, z, h, box, cstone::sfcKindPointer(particleKeys), neighbors, &neighborsCount, numParticles, ngmax);

        alpha[i] = kernels::AVswitchesJLoop(i,
                                            sincIndex,
                                            K,
                                            box,
                                            neighbors,
                                            neighborsCount,
                                            x,
                                            y,
                                            z,
                                            vx,
                                            vy,
                                            vz,
                                            h,
                                            m,
                                            c,
                                            c11,
                                            c12,
                                            c13,
                                            c22,
                                            c23,
                                            c33,
                                            wh,
                                            whd,
                                            kx,
                                            rho0,
                                            divv,
                                            dt,
                                            alphamin,
                                            alphamax,
                                            decay_constant,
                                            alpha[i]);
    }
}

template<class Dataset>
void computeAVswitches(size_t startIndex, size_t endIndex, size_t ngmax, Dataset& d,
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

    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_vx, d.vx.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_vy, d.vy.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_vz, d.vz.data(), size_np_T, cudaMemcpyHostToDevice));

    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_h, d.h.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_m, d.m.data(), size_np_T, cudaMemcpyHostToDevice));

    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_c11, d.c11.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_c12, d.c12.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_c13, d.c13.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_c22, d.c22.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_c23, d.c23.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_c33, d.c33.data(), size_np_T, cudaMemcpyHostToDevice));

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

        cudaAVswitches<<<numBlocks, numThreads, 0, stream>>>(d.sincIndex,
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
                                                             d.devPtrs.d_vx,
                                                             d.devPtrs.d_vy,
                                                             d.devPtrs.d_vz,
                                                             d.devPtrs.d_h,
                                                             d.devPtrs.d_m,
                                                             d.devPtrs.d_c,
                                                             d.devPtrs.d_c11,
                                                             d.devPtrs.d_c12,
                                                             d.devPtrs.d_c13,
                                                             d.devPtrs.d_c22,
                                                             d.devPtrs.d_c23,
                                                             d.devPtrs.d_c33,
                                                             d.devPtrs.d_wh,
                                                             d.devPtrs.d_whd,
                                                             d.devPtrs.d_kx,
                                                             d.devPtrs.d_rho0,
                                                             d.devPtrs.d_divv,
                                                             d.minDt,
                                                             d.alphamin,
                                                             d.alphamax,
                                                             d.decay_constant,
                                                             d.devPtrs.d_alpha);

        CHECK_CUDA_ERR(cudaGetLastError());
    }

    // Memcpy in default stream synchronizes all other streams
    CHECK_CUDA_ERR(cudaMemcpy(d.alpha.data(), d.devPtrs.d_alpha, size_np_T, cudaMemcpyDeviceToHost));
}

template void computeAVswitches(size_t, size_t, size_t, ParticlesData<double, unsigned, cstone::GpuTag>&,
                                const cstone::Box<double>&);
template void computeAVswitches(size_t, size_t, size_t, ParticlesData<double, uint64_t, cstone::GpuTag>&,
                                const cstone::Box<double>&);
template void computeAVswitches(size_t, size_t, size_t, ParticlesData<float, unsigned, cstone::GpuTag>&,
                                const cstone::Box<float>&);
template void computeAVswitches(size_t, size_t, size_t, ParticlesData<float, uint64_t, cstone::GpuTag>&,
                                const cstone::Box<float>&);

} // namespace cuda
} // namespace sph
} // namespace sphexa
