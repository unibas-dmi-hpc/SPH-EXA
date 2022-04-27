#include "sph.cuh"

#include "sph/kernel/momentum_energy_kern.hpp"

namespace sphexa
{
namespace sph
{
namespace cuda
{

//! @brief compute atomic min for floats using integer operations
__device__ __forceinline__ float atomicMinFloat(float* addr, float value)
{
    float old;
    old = (value >= 0) ? __int_as_float(atomicMin((int*)addr, __float_as_int(value)))
                       : __uint_as_float(atomicMax((unsigned int*)addr, __float_as_uint(value)));

    return old;
}

__device__ float minDt_device;

/*! @brief
 *
 * @tparam     T               float or double
 * @tparam     KeyType         32- or 64-bit unsigned integer
 * @param[in]  sincIndex
 * @param[in]  K
 * @param[in]  Kcour
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
 * @param[in]  rho
 * @param[in]  p
 * @param[in]  c
 * @param[in]  c11             IAD components, length @p numParticles
 * @param[in]  c12
 * @param[in]  c13
 * @param[in]  c22
 * @param[in]  c23
 * @param[in]  c33
 * @param[in]  Atmin
 * @param[in]  Atmax
 * @param[in]  ramp
 * @param[in]  wh              sinc lookup table
 * @param[in]  whd             sinc derivative lookup table
 * @param[in]  kx
 * @param[in]  rho0
 * @param[in]  alpha
 * @param[out] grad_P_x
 * @param[out] grad_P_y
 * @param[out] grad_P_z
 * @param[out] du
 *
 */
template<class T, class KeyType>
__global__ void cudaMomentumEnergy(
        T sincIndex,
        T K,
        T Kcour,
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
        const T* rho,
        const T* p,
        const T* c,
        const T* c11,
        const T* c12,
        const T* c13,
        const T* c22,
        const T* c23,
        const T* c33,
        const T  Atmin,
        const T  Atmax,
        const T  ramp,
        const T* wh,
        const T* whd,
        const T* kx,
        const T* rho0,
        const T* alpha,
        T* grad_P_x,
        T* grad_P_y,
        T* grad_P_z,
        T* du)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned i   = tid + firstParticle;

    // need to hard-code ngmax stack allocation for now
    assert(ngmax <= CudaConfig::NGMAX && "ngmax too big, please increase NGMAX to desired value");
    int neighbors[CudaConfig::NGMAX];
    int neighborsCount;

    // starting from CUDA 11.3, dynamic stack allocation is available with the following command
    // int* neighbors = (int*)alloca(ngmax * sizeof(int));

    T dt_i = INFINITY;

    if (i < lastParticle)
    {
        cstone::findNeighbors(
            i, x, y, z, h, box, cstone::sfcKindPointer(particleKeys), neighbors, &neighborsCount, numParticles, ngmax);

        T maxvsignal;
        sph::kernels::momentumEnergyJLoop(i,
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
                                          rho,
                                          p,
                                          c,
                                          c11,
                                          c12,
                                          c13,
                                          c22,
                                          c23,
                                          c33,
                                          Atmin,
                                          Atmax,
                                          ramp,
                                          wh,
                                          whd,
                                          kx,
                                          rho0,
                                          alpha,
                                          grad_P_x,
                                          grad_P_y,
                                          grad_P_z,
                                          du,
                                          &maxvsignal);

        dt_i = sph::kernels::tsKCourant(maxvsignal, h[i], c[i], Kcour);
    }

    typedef cub::BlockReduce<T, CudaConfig::numThreads> BlockReduce;
    __shared__ typename BlockReduce::TempStorage        temp_storage;

    BlockReduce reduce(temp_storage);
    T           blockMin = reduce.Reduce(dt_i, cub::Min());
    __syncthreads();

    if (threadIdx.x == 0) { atomicMinFloat(&minDt_device, blockMin); }
}

template<class Dataset>
void computeMomentumEnergy(size_t startIndex, size_t endIndex, size_t ngmax, Dataset& d,
                           const cstone::Box<typename Dataset::RealType>& box)
{
    using T = typename Dataset::RealType;

    size_t sizeWithHalos = d.x.size();
    size_t size_np_T     = sizeWithHalos * sizeof(T);

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

    unsigned numParticlesCompute = endIndex - startIndex;

    unsigned numThreads = CudaConfig::numThreads;
    unsigned numBlocks  = (numParticlesCompute + numThreads - 1) / numThreads;

    float huge = 1e10;
    CHECK_CUDA_ERR(cudaMemcpyToSymbol(minDt_device, &huge, sizeof(huge)));

    cudaMomentumEnergy<<<numBlocks, numThreads>>>(d.sincIndex,
                                                  d.K,
                                                  d.Kcour,
                                                  ngmax,
                                                  box,
                                                  startIndex,
                                                  endIndex,
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
                                                  d.devPtrs.d_rho,
                                                  d.devPtrs.d_p,
                                                  d.devPtrs.d_c,
                                                  d.devPtrs.d_c11,
                                                  d.devPtrs.d_c12,
                                                  d.devPtrs.d_c13,
                                                  d.devPtrs.d_c22,
                                                  d.devPtrs.d_c23,
                                                  d.devPtrs.d_c33,
                                                  d.Atmin,
                                                  d.Atmax,
                                                  d.ramp,
                                                  d.devPtrs.d_wh,
                                                  d.devPtrs.d_whd,
                                                  d.devPtrs.d_kx,
                                                  d.devPtrs.d_rho0,
                                                  d.devPtrs.d_alpha,
                                                  d.devPtrs.d_grad_P_x,
                                                  d.devPtrs.d_grad_P_y,
                                                  d.devPtrs.d_grad_P_z,
                                                  d.devPtrs.d_du);

    CHECK_CUDA_ERR(cudaGetLastError());

    float minDt;
    CHECK_CUDA_ERR(cudaMemcpyFromSymbol(&minDt, minDt_device, sizeof(minDt)));
    d.minDt_loc = minDt;
}

template void computeMomentumEnergy(size_t, size_t, size_t, ParticlesData<double, unsigned, cstone::GpuTag>& d,
                                    const cstone::Box<double>&);
template void computeMomentumEnergy(size_t, size_t, size_t, ParticlesData<double, uint64_t, cstone::GpuTag>& d,
                                    const cstone::Box<double>&);
template void computeMomentumEnergy(size_t, size_t, size_t, ParticlesData<float, unsigned, cstone::GpuTag>& d,
                                    const cstone::Box<float>&);
template void computeMomentumEnergy(size_t, size_t, size_t, ParticlesData<float, uint64_t, cstone::GpuTag>& d,
                                    const cstone::Box<float>&);

} // namespace cuda
} // namespace sph
} // namespace sphexa
