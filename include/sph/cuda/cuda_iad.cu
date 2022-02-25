#include <algorithm>

#include "sph.cuh"
#include "particles_data.hpp"
#include "cuda_utils.cuh"
#include "sph/kernel/iad.hpp"

#include "cstone/cuda/findneighbors.cuh"

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
 * @param[in]  ro              densities, length @p numParticles
 * @param[in]  wh              sinc lookup table
 * @param[in]  whd             sinc derivative lookup table
 * @param[out] c11             output IAD components, length @p numParticles
 * @param[out] c12
 * @param[out] c13
 * @param[out] c22
 * @param[out] c23
 * @param[out] c33
 */
template<class T, class KeyType>
__global__ void computeIAD(T sincIndex, T K, int ngmax, cstone::Box<T> box, int firstParticle, int lastParticle,
                           int numParticles, const KeyType* particleKeys, const T* x, const T* y, const T* z,
                           const T* h, const T* m, const T* ro, const T* wh, const T* whd, T* c11, T* c12, T* c13,
                           T* c22, T* c23, T* c33)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned i   = tid + firstParticle;

    if (i >= lastParticle) return;

    // need to hard-code ngmax stack allocation for now
    assert(ngmax <= NGMAX && "ngmax too big, please increase NGMAX to desired size");
    int neighbors[NGMAX];
    int neighborsCount;

    // starting from CUDA 11.3, dynamic stack allocation is available with the following command
    // int* neighbors = (int*)alloca(ngmax * sizeof(int));

    cstone::findNeighbors(
        i, x, y, z, h, box, cstone::sfcKindPointer(particleKeys), neighbors, &neighborsCount, numParticles, ngmax);

    sph::kernels::IADJLoop(
        i, sincIndex, K, box, neighbors, neighborsCount, x, y, z, h, m, ro, wh, whd, c11, c12, c13, c22, c23, c33);
}

template<class Dataset>
void computeIAD(const std::vector<Task>& taskList, Dataset& d, const cstone::Box<double>& box)
{
    using T = typename Dataset::RealType;

    // number of locally present particles, including halos
    size_t numParticles = d.x.size();

    size_t size_np_T = numParticles * sizeof(T);
    T      ngmax     = taskList.empty() ? 0 : taskList.front().ngmax;

    CHECK_CUDA_ERR(cudaMemcpy(d.devPtrs.d_ro, d.ro.data(), size_np_T, cudaMemcpyHostToDevice));

    unsigned firstParticle       = taskList.front().firstParticle;
    unsigned lastParticle        = taskList.back().lastParticle;
    unsigned numParticlesCompute = lastParticle - firstParticle;

    unsigned numThreads = 128;
    unsigned numBlocks  = (numParticlesCompute + numThreads - 1) / numThreads;

    computeIAD<<<numBlocks, numThreads>>>(d.sincIndex,
                                          d.K,
                                          ngmax,
                                          box,
                                          firstParticle,
                                          lastParticle,
                                          numParticles,
                                          d.devPtrs.d_codes,
                                          d.devPtrs.d_x,
                                          d.devPtrs.d_y,
                                          d.devPtrs.d_z,
                                          d.devPtrs.d_h,
                                          d.devPtrs.d_m,
                                          d.devPtrs.d_ro,
                                          d.devPtrs.d_wh,
                                          d.devPtrs.d_whd,
                                          d.devPtrs.d_c11,
                                          d.devPtrs.d_c12,
                                          d.devPtrs.d_c13,
                                          d.devPtrs.d_c22,
                                          d.devPtrs.d_c23,
                                          d.devPtrs.d_c33);
    CHECK_CUDA_ERR(cudaGetLastError());

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
