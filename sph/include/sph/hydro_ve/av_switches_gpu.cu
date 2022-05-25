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
 * @brief Density i-loop GPU driver
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "cstone/cuda/findneighbors.cuh"

#include "sph/sph.cuh"
#include "sph/particles_data.hpp"
#include "sph/util/cuda_utils.cuh"
#include "sph/hydro_ve/av_switches_kern.hpp"

namespace sph
{
namespace cuda
{

template<typename T, class KeyType>
__global__ void AVswitchesGpu(T sincIndex, T K, int ngmax, const cstone::Box<T> box, size_t first, size_t last,
                              size_t numParticles, const KeyType* particleKeys, const T* x, const T* y, const T* z,
                              const T* vx, const T* vy, const T* vz, const T* h, const T* c, const T* c11, const T* c12,
                              const T* c13, const T* c22, const T* c23, const T* c33, const T* wh, const T* whd,
                              const T* kx, const T* xm, const T* divv, T minDt, T alphamin, T alphamax,
                              T decay_constant, T* alpha)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned i   = tid + first;

    if (i >= last) return;

    // need to hard-code ngmax stack allocation for now
    assert(ngmax <= NGMAX && "ngmax too big, please increase NGMAX to desired size");
    int neighbors[NGMAX];
    int neighborsCount;

    // starting from CUDA 11.3, dynamic stack allocation is available with the following command
    // int* neighbors = (int*)alloca(ngmax * sizeof(int));

    cstone::findNeighbors(
        i, x, y, z, h, box, cstone::sfcKindPointer(particleKeys), neighbors, &neighborsCount, numParticles, ngmax);

    alpha[i] = AVswitchesJLoop(i,
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
                               xm,
                               divv,
                               minDt,
                               alphamin,
                               alphamax,
                               decay_constant,
                               alpha[i]);
}

template<class Dataset>
void computeAVswitches(size_t startIndex, size_t endIndex, size_t ngmax, Dataset& d,
                       const cstone::Box<typename Dataset::RealType>& box)
{
    // number of locally present particles, including halos
    size_t sizeWithHalos       = d.x.size();
    size_t numParticlesCompute = endIndex - startIndex;

    unsigned numThreads = 128;
    unsigned numBlocks  = (numParticlesCompute + numThreads - 1) / numThreads;

    AVswitchesGpu<<<numBlocks, numThreads>>>(d.sincIndex,
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
                                             rawPtr(d.devData.vx),
                                             rawPtr(d.devData.vy),
                                             rawPtr(d.devData.vz),
                                             rawPtr(d.devData.h),
                                             rawPtr(d.devData.c),
                                             rawPtr(d.devData.c11),
                                             rawPtr(d.devData.c12),
                                             rawPtr(d.devData.c13),
                                             rawPtr(d.devData.c22),
                                             rawPtr(d.devData.c23),
                                             rawPtr(d.devData.c33),
                                             rawPtr(d.devData.wh),
                                             rawPtr(d.devData.whd),
                                             rawPtr(d.devData.kx),
                                             rawPtr(d.devData.xm),
                                             rawPtr(d.devData.divv),
                                             d.minDt,
                                             d.alphamin,
                                             d.alphamax,
                                             d.decay_constant,
                                             rawPtr(d.devData.alpha));
    CHECK_CUDA_ERR(cudaGetLastError());
}

template void computeAVswitches(size_t, size_t, size_t, sphexa::ParticlesData<double, unsigned, cstone::GpuTag>& d,
                                const cstone::Box<double>&);
template void computeAVswitches(size_t, size_t, size_t, sphexa::ParticlesData<double, uint64_t, cstone::GpuTag>& d,
                                const cstone::Box<double>&);
template void computeAVswitches(size_t, size_t, size_t, sphexa::ParticlesData<float, unsigned, cstone::GpuTag>& d,
                                const cstone::Box<float>&);
template void computeAVswitches(size_t, size_t, size_t, sphexa::ParticlesData<float, uint64_t, cstone::GpuTag>& d,
                                const cstone::Box<float>&);

} // namespace cuda
} // namespace sph
