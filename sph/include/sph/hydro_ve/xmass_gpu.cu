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

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/findneighbors.hpp"

#include "sph/sph_gpu.hpp"
#include "sph/particles_data.hpp"
#include "sph/hydro_ve/xmass_kern.hpp"

namespace sph
{
namespace cuda
{

template<typename T, class KeyType>
__global__ void xmassGpu(T sincIndex, T K, unsigned ngmax, const cstone::Box<T> box, size_t first, size_t last,
                         size_t numParticles, const KeyType* particleKeys, unsigned* neighborsCount, const T* x,
                         const T* y, const T* z, const T* h, const T* m, const T* wh, const T* whd, T* xm)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned i   = tid + first;

    if (i >= last) return;

    // need to hard-code ngmax stack allocation for now
    assert(ngmax <= NGMAX && "ngmax too big, please increase NGMAX to desired size");
    cstone::LocalIndex neighbors[NGMAX];
    unsigned           neighborsCount_;

    // starting from CUDA 11.3, dynamic stack allocation is available with the following command
    // int* neighbors = (int*)alloca(ngmax * sizeof(int));

    cstone::findNeighbors(i, x, y, z, h, box, cstone::sfcKindPointer(particleKeys), neighbors, &neighborsCount_,
                          numParticles, ngmax);
    unsigned nc = stl::min(neighborsCount_, ngmax);

    xm[i] = sph::xmassJLoop(i, sincIndex, K, box, neighbors, nc, x, y, z, h, m, wh, whd);

    neighborsCount[tid] = neighborsCount_;
}

template<class Dataset>
void computeXMass(size_t startIndex, size_t endIndex, unsigned ngmax, Dataset& d,
                  const cstone::Box<typename Dataset::RealType>& box)
{
    using T       = typename Dataset::RealType;
    using KeyType = typename Dataset::KeyType;

    size_t sizeWithHalos     = d.x.size();
    size_t numLocalParticles = endIndex - startIndex;

    size_t taskSize = sphexa::DeviceParticlesData<T, KeyType>::taskSize;
    size_t numTasks = iceil(numLocalParticles, taskSize);

    // number of CUDA streams to use
    constexpr int NST = sphexa::DeviceParticlesData<T, Dataset>::NST;

    for (int i = 0; i < numTasks; ++i)
    {
        int          sIdx   = i % NST;
        cudaStream_t stream = d.devData.d_stream[sIdx].stream;

        unsigned* d_neighborsCount_use = d.devData.d_stream[sIdx].d_neighborsCount;

        size_t firstParticle       = startIndex + i * taskSize;
        size_t lastParticle        = std::min(startIndex + (i + 1) * taskSize, endIndex);
        size_t numParticlesCompute = lastParticle - firstParticle;

        unsigned numThreads = 256;
        unsigned numBlocks  = (numParticlesCompute + numThreads - 1) / numThreads;

        xmassGpu<<<numBlocks, numThreads, 0, stream>>>(
            d.sincIndex, d.K, ngmax, box, firstParticle, lastParticle, sizeWithHalos, rawPtr(d.devData.keys),
            d_neighborsCount_use, rawPtr(d.devData.x), rawPtr(d.devData.y), rawPtr(d.devData.z), rawPtr(d.devData.h),
            rawPtr(d.devData.m), rawPtr(d.devData.wh), rawPtr(d.devData.whd), rawPtr(d.devData.xm));

        checkGpuErrors(cudaMemcpyAsync(d.nc.data() + firstParticle, d_neighborsCount_use,
                                       numParticlesCompute * sizeof(decltype(d.nc.front())),
                                       cudaMemcpyDeviceToHost, stream));
    }
    checkGpuErrors(cudaDeviceSynchronize());
}

template void computeXMass(size_t, size_t, unsigned, sphexa::ParticlesData<double, unsigned, cstone::GpuTag>& d,
                           const cstone::Box<double>&);
template void computeXMass(size_t, size_t, unsigned, sphexa::ParticlesData<double, uint64_t, cstone::GpuTag>& d,
                           const cstone::Box<double>&);
template void computeXMass(size_t, size_t, unsigned, sphexa::ParticlesData<float, unsigned, cstone::GpuTag>& d,
                           const cstone::Box<float>&);
template void computeXMass(size_t, size_t, unsigned, sphexa::ParticlesData<float, uint64_t, cstone::GpuTag>& d,
                           const cstone::Box<float>&);

} // namespace cuda
} // namespace sph
