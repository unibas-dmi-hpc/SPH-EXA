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
#include "cstone/traversal/find_neighbors.cuh"

#include "sph/sph_gpu.hpp"
#include "sph/particles_data.hpp"
#include "sph/hydro_ve/iad_kern.hpp"
#include "sph/hydro_ve/divv_curlv_kern.hpp"

namespace sph
{
namespace cuda
{

using cstone::GpuConfig;
using cstone::TravConfig;
using cstone::TreeNodeIndex;

template<class Tc, class T, class KeyType>
__global__ void iadDivvCurlvGpu(T sincIndex, T K, unsigned ngmax, const cstone::Box<Tc> box, size_t first, size_t last,
                                const cstone::OctreeNsView<Tc, KeyType> tree, const Tc* x, const Tc* y, const Tc* z,
                                const T* vx, const T* vy, const T* vz, const T* h, const T* wh, const T* whd,
                                const T* xm, const T* kx, T* c11, T* c12, T* c13, T* c22, T* c23, T* c33, T* divv,
                                T* curlv, T* dV11, T* dV12, T* dV13, T* dV22, T* dV23, T* dV33,
                                cstone::LocalIndex* nidx, TreeNodeIndex* globalPool, bool doGradV)
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

        if (targetIdx >= numTargets) return;

        cstone::LocalIndex bodyBegin = first + targetIdx * TravConfig::targetSize;
        cstone::LocalIndex bodyEnd   = cstone::imin(bodyBegin + TravConfig::targetSize, last);
        cstone::LocalIndex i         = bodyBegin + laneIdx;

        auto ncTrue = traverseNeighbors(bodyBegin, bodyEnd, x, y, z, h, tree, box, neighborsWarp, ngmax, globalPool);

        if (i >= last) continue;

        unsigned ncCapped = stl::min(ncTrue[0], ngmax);
        IADJLoop<TravConfig::targetSize>(i, sincIndex, K, box, neighborsWarp + laneIdx, ncCapped, x, y, z, h, wh, whd,
                                         xm, kx, c11, c12, c13, c22, c23, c33);
        divV_curlVJLoop<TravConfig::targetSize>(i, sincIndex, K, box, neighborsWarp + laneIdx, ncCapped, x, y, z, vx,
                                                vy, vz, h, c11, c12, c13, c22, c23, c33, wh, whd, kx, xm, divv, curlv,
                                                dV11, dV12, dV13, dV22, dV23, dV33, doGradV);
    }
}

template<class Dataset>
void computeIadDivvCurlv(size_t startIndex, size_t endIndex, unsigned ngmax, Dataset& d,
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

    bool doGradV = d.devData.x.size() == d.devData.dV11.size();

    iadDivvCurlvGpu<<<numBlocks, TravConfig::numThreads>>>(
        d.sincIndex, d.K, ngmax, box, startIndex, endIndex, d.treeView, rawPtr(d.devData.x), rawPtr(d.devData.y),
        rawPtr(d.devData.z), rawPtr(d.devData.vx), rawPtr(d.devData.vy), rawPtr(d.devData.vz), rawPtr(d.devData.h),
        rawPtr(d.devData.wh), rawPtr(d.devData.whd), rawPtr(d.devData.xm), rawPtr(d.devData.kx), rawPtr(d.devData.c11),
        rawPtr(d.devData.c12), rawPtr(d.devData.c13), rawPtr(d.devData.c22), rawPtr(d.devData.c23),
        rawPtr(d.devData.c33), rawPtr(d.devData.divv), rawPtr(d.devData.curlv), rawPtr(d.devData.dV11),
        rawPtr(d.devData.dV12), rawPtr(d.devData.dV13), rawPtr(d.devData.dV22), rawPtr(d.devData.dV23),
        rawPtr(d.devData.dV33), nidxPool, traversalPool, doGradV);
    checkGpuErrors(cudaDeviceSynchronize());
}

#define IAD_DIVV_CURLV(real, key)                                                                                      \
    template void computeIadDivvCurlv(size_t, size_t, unsigned, sphexa::ParticlesData<real, key, cstone::GpuTag>& d,   \
                                      const cstone::Box<real>&)

IAD_DIVV_CURLV(double, uint32_t);
IAD_DIVV_CURLV(double, uint64_t);
IAD_DIVV_CURLV(float, uint32_t);
IAD_DIVV_CURLV(float, uint64_t);

} // namespace cuda
} // namespace sph
