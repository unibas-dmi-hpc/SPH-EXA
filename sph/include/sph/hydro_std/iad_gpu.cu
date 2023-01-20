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

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/findneighbors.hpp"
#include "cstone/traversal/find_neighbors.cuh"

#include "sph/sph_gpu.hpp"
#include "sph/particles_data.hpp"
#include "sph/hydro_std/iad_kern.hpp"

namespace sph
{

using cstone::GpuConfig;
using cstone::LocalIndex;
using cstone::TravConfig;
using cstone::TreeNodeIndex;

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
template<class Tc, class Tm, class T, class KeyType>
__global__ void IADGpuKernel(T sincIndex, T K, unsigned ngmax, cstone::Box<T> box, size_t first, size_t last,
                             const cstone::OctreeNsView<Tc, KeyType> tree, const Tc* x, const Tc* y, const Tc* z,
                             const T* h, const Tm* m, const T* rho, const T* wh, const T* whd, T* c11, T* c12, T* c13,
                             T* c22, T* c23, T* c33, LocalIndex* nidx, TreeNodeIndex* globalPool)
{
    unsigned laneIdx     = threadIdx.x & (GpuConfig::warpSize - 1);
    unsigned numTargets  = (last - first - 1) / TravConfig::targetSize + 1;
    unsigned targetIdx   = 0;
    unsigned warpIdxGrid = (blockDim.x * blockIdx.x + threadIdx.x) >> GpuConfig::warpSizeLog2;

    cstone::LocalIndex* neighborsWarp = nidx + ngmax * TravConfig::targetSize * warpIdxGrid;

    while (true)
    {
        // first thread in warp grabs next target
        if (laneIdx == 0) { targetIdx = atomicAdd(&cstone::targetCounterGlob, 1); }
        targetIdx = cstone::shflSync(targetIdx, 0);

        if (targetIdx >= numTargets) { return; }

        cstone::LocalIndex bodyBegin = first + targetIdx * TravConfig::targetSize;
        cstone::LocalIndex bodyEnd   = cstone::imin(bodyBegin + TravConfig::targetSize, last);
        cstone::LocalIndex i         = bodyBegin + laneIdx;

        auto ncTrue = traverseNeighbors(bodyBegin, bodyEnd, x, y, z, h, tree, box, neighborsWarp, ngmax, globalPool);

        if (i >= last) { continue; }

        unsigned ncCapped = stl::min(ncTrue[0], ngmax);
        sph::IADJLoopSTD<TravConfig::targetSize>(i, sincIndex, K, box, neighborsWarp + laneIdx, ncCapped, x, y, z, h, m,
                                                 rho, wh, whd, c11, c12, c13, c22, c23, c33);
    }
}

template<class Dataset>
void computeIADGpu(size_t startIndex, size_t endIndex, unsigned ngmax, Dataset& d,
                   const cstone::Box<typename Dataset::RealType>& box)
{
    unsigned numWarpsPerBlock = TravConfig::numThreads / GpuConfig::warpSize;
    unsigned numBodies        = endIndex - startIndex;
    unsigned numWarps         = (numBodies - 1) / TravConfig::targetSize + 1;
    unsigned numBlocks        = (numWarps - 1) / numWarpsPerBlock + 1;
    numBlocks                 = std::min(numBlocks, TravConfig::maxNumActiveBlocks);

    unsigned poolSize = TravConfig::memPerWarp * numWarpsPerBlock * numBlocks;
    unsigned nidxSize = ngmax * numBlocks * TravConfig::numThreads;
    reallocateDestructive(d.devData.traversalStack, poolSize + nidxSize, 1.01);
    auto* traversalPool = reinterpret_cast<TreeNodeIndex*>(rawPtr(d.devData.traversalStack));
    auto* nidxPool      = rawPtr(d.devData.traversalStack) + poolSize;

    cstone::resetTraversalCounters<<<1, 1>>>();

    IADGpuKernel<<<numBlocks, TravConfig::numThreads>>>(
        d.sincIndex, d.K, ngmax, box, startIndex, endIndex, d.treeView, rawPtr(d.devData.x), rawPtr(d.devData.y),
        rawPtr(d.devData.z), rawPtr(d.devData.h), rawPtr(d.devData.m), rawPtr(d.devData.rho), rawPtr(d.devData.wh),
        rawPtr(d.devData.whd), rawPtr(d.devData.c11), rawPtr(d.devData.c12), rawPtr(d.devData.c13),
        rawPtr(d.devData.c22), rawPtr(d.devData.c23), rawPtr(d.devData.c33), nidxPool, traversalPool);
    checkGpuErrors(cudaDeviceSynchronize());
}

template void computeIADGpu(size_t, size_t, unsigned, sphexa::ParticlesData<double, unsigned, cstone::GpuTag>& d,
                            const cstone::Box<double>&);
template void computeIADGpu(size_t, size_t, unsigned, sphexa::ParticlesData<double, uint64_t, cstone::GpuTag>& d,
                            const cstone::Box<double>&);
template void computeIADGpu(size_t, size_t, unsigned, sphexa::ParticlesData<float, unsigned, cstone::GpuTag>& d,
                            const cstone::Box<float>&);
template void computeIADGpu(size_t, size_t, unsigned, sphexa::ParticlesData<float, uint64_t, cstone::GpuTag>& d,
                            const cstone::Box<float>&);

} // namespace sph
