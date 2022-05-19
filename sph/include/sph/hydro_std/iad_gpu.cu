/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief Integral-approach-to-derivative i-loop GPU driver
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "sph/sph.cuh"
#include "sph/particles_data.hpp"
#include "sph/util/cuda_utils.cuh"
#include "sph/hydro_std/iad_kern.hpp"

#include "cstone/cuda/findneighbors.cuh"

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
 * @param[in]  rho             densities, length @p numParticles
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
__global__ void cudaIAD(T sincIndex, T K, int ngmax, cstone::Box<T> box, int firstParticle, int lastParticle,
                        int numParticles, const KeyType* particleKeys, const T* x, const T* y, const T* z, const T* h,
                        const T* m, const T* rho, const T* wh, const T* whd, T* c11, T* c12, T* c13, T* c22, T* c23,
                        T* c33)
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

    sph::IADJLoopSTD(
        i, sincIndex, K, box, neighbors, neighborsCount, x, y, z, h, m, rho, wh, whd, c11, c12, c13, c22, c23, c33);
}

template<class Dataset>
void computeIAD(size_t startIndex, size_t endIndex, size_t ngmax, Dataset& d,
                const cstone::Box<typename Dataset::RealType>& box)
{
    using T = typename Dataset::RealType;

    // number of locally present particles, including halos
    size_t sizeWithHalos = d.x.size();

    unsigned numParticlesCompute = endIndex - startIndex;

    unsigned numThreads = 128;
    unsigned numBlocks  = (numParticlesCompute + numThreads - 1) / numThreads;

    cudaIAD<<<numBlocks, numThreads>>>(d.sincIndex,
                                       d.K,
                                       ngmax,
                                       box,
                                       startIndex,
                                       endIndex,
                                       sizeWithHalos,
                                       rawPtr(d.devData.codes),
                                       rawPtr(d.devData.x),
                                       rawPtr(d.devData.y),
                                       rawPtr(d.devData.z),
                                       rawPtr(d.devData.h),
                                       rawPtr(d.devData.m),
                                       rawPtr(d.devData.rho),
                                       rawPtr(d.devData.wh),
                                       rawPtr(d.devData.whd),
                                       rawPtr(d.devData.c11),
                                       rawPtr(d.devData.c12),
                                       rawPtr(d.devData.c13),
                                       rawPtr(d.devData.c22),
                                       rawPtr(d.devData.c23),
                                       rawPtr(d.devData.c33));
    CHECK_CUDA_ERR(cudaGetLastError());
}

template void computeIAD(size_t, size_t, size_t, sphexa::ParticlesData<double, unsigned, cstone::GpuTag>& d,
                         const cstone::Box<double>&);
template void computeIAD(size_t, size_t, size_t, sphexa::ParticlesData<double, uint64_t, cstone::GpuTag>& d,
                         const cstone::Box<double>&);
template void computeIAD(size_t, size_t, size_t, sphexa::ParticlesData<float, unsigned, cstone::GpuTag>& d,
                         const cstone::Box<float>&);
template void computeIAD(size_t, size_t, size_t, sphexa::ParticlesData<float, uint64_t, cstone::GpuTag>& d,
                         const cstone::Box<float>&);

} // namespace cuda
} // namespace sph
