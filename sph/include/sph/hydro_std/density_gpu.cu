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

#include <algorithm>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/findneighbors.hpp"

#include "sph/sph_gpu.hpp"
#include "sph/particles_data.hpp"
#include "sph/hydro_std/density_kern.hpp"

namespace sph
{
namespace cuda
{

template<class Tc, class Tm, class T, class KeyType>
__global__ void cudaDensity(T sincIndex, T K, unsigned ngmax, cstone::Box<T> box, size_t firstParticle,
                            size_t lastParticle, size_t numParticles, const KeyType* particleKeys, unsigned* nc,
                            const Tc* x, const Tc* y, const Tc* z, const T* h, const Tm* m, const T* wh, const T* whd,
                            T* rho)
{
    cstone::LocalIndex tid = blockDim.x * blockIdx.x + threadIdx.x;
    cstone::LocalIndex i   = tid + firstParticle;
    if (i >= lastParticle) return;

    // need to hard-code ngmax stack allocation for now
    assert(ngmax <= NGMAX && "ngmax too big, please increase NGMAX to desired value");
    cstone::LocalIndex neighbors[NGMAX];
    unsigned           ncTrue;

    // starting from CUDA 11.3, dynamic stack allocation is available with the following command
    // int* neighbors = (int*)alloca(ngmax * sizeof(int));

    cstone::findNeighbors(i, x, y, z, h, box, cstone::sfcKindPointer(particleKeys), neighbors, &ncTrue, numParticles,
                          ngmax);

    unsigned ncCapped = stl::min(ncTrue, ngmax);
    rho[i]            = sph::densityJLoop(i, sincIndex, K, box, neighbors, ncCapped, x, y, z, h, m, wh, whd);
    nc[i]             = ncTrue;
}

template<class Dataset>
void computeDensity(size_t startIndex, size_t endIndex, unsigned ngmax, Dataset& d,
                    const cstone::Box<typename Dataset::RealType>& box)
{
    using T       = typename Dataset::RealType;
    using KeyType = typename Dataset::KeyType;

    size_t numParticles  = endIndex - startIndex;
    size_t sizeWithHalos = d.devData.x.size();

    unsigned numThreads = 256;
    unsigned numBlocks  = (numParticles + numThreads - 1) / numThreads;

    cudaDensity<<<numBlocks, numThreads>>>(
        d.sincIndex, d.K, ngmax, box, startIndex, endIndex, sizeWithHalos, rawPtr(d.devData.keys), rawPtr(d.devData.nc),
        rawPtr(d.devData.x), rawPtr(d.devData.y), rawPtr(d.devData.z), rawPtr(d.devData.h), rawPtr(d.devData.m),
        rawPtr(d.devData.wh), rawPtr(d.devData.whd), rawPtr(d.devData.rho));
    checkGpuErrors(cudaDeviceSynchronize());
}

template void computeDensity(size_t, size_t, unsigned, sphexa::ParticlesData<double, unsigned, cstone::GpuTag>&,
                             const cstone::Box<double>&);
template void computeDensity(size_t, size_t, unsigned, sphexa::ParticlesData<double, uint64_t, cstone::GpuTag>&,
                             const cstone::Box<double>&);
template void computeDensity(size_t, size_t, unsigned, sphexa::ParticlesData<float, unsigned, cstone::GpuTag>&,
                             const cstone::Box<float>&);
template void computeDensity(size_t, size_t, unsigned, sphexa::ParticlesData<float, uint64_t, cstone::GpuTag>&,
                             const cstone::Box<float>&);

} // namespace cuda
} // namespace sph
