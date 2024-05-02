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

#include <thrust/transform.h>

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/findneighbors.hpp"
#include "cstone/traversal/find_neighbors.cuh"
#include "cstone/traversal/groups.cuh"

#include "sph/sph_gpu.hpp"
#include "sph/particles_data.hpp"
#include "sph/hydro_ve/xmass_kern.hpp"

namespace sph
{

using cstone::GpuConfig;
using cstone::NcStats;
using cstone::TravConfig;
using cstone::TreeNodeIndex;

namespace cuda
{

__device__ bool nc_h_convergenceFailure = false;

template<class Tc, class Tm, class T, class KeyType>
__global__ void xmassGpu(Tc K, unsigned ng0, unsigned ngmax, const cstone::Box<Tc> box,
                         const cstone::LocalIndex* groups, cstone::LocalIndex numGroups,
                         const cstone::OctreeNsView<Tc, KeyType> tree, unsigned* nc, const Tc* x, const Tc* y,
                         const Tc* z, T* h, const Tm* m, const T* wh, const T* whd, T* xm, cstone::LocalIndex* nidx,
                         TreeNodeIndex* globalPool)
{
    unsigned laneIdx     = threadIdx.x & (GpuConfig::warpSize - 1);
    unsigned targetIdx   = 0;
    unsigned warpIdxGrid = (blockDim.x * blockIdx.x + threadIdx.x) >> GpuConfig::warpSizeLog2;

    cstone::LocalIndex* neighborsWarp = nidx + ngmax * TravConfig::targetSize * warpIdxGrid;

    while (true)
    {
        // first thread in warp grabs next target
        if (laneIdx == 0) { targetIdx = atomicAdd(&cstone::targetCounterGlob, 1); }
        targetIdx = cstone::shflSync(targetIdx, 0);

        if (targetIdx >= numGroups) return;

        cstone::LocalIndex bodyBegin = groups[targetIdx];
        cstone::LocalIndex bodyEnd   = groups[targetIdx + 1];
        cstone::LocalIndex i         = bodyBegin + laneIdx;

        unsigned ncSph =
            1 + traverseNeighbors(bodyBegin, bodyEnd, x, y, z, h, tree, box, neighborsWarp, ngmax, globalPool)[0];

        constexpr int ncMaxIteration = 9;
        for (int ncIt = 0; ncIt <= ncMaxIteration; ++ncIt)
        {
            bool repeat = (ncSph < ng0 / 4 || (ncSph - 1) > ngmax) && i < bodyEnd;
            if (!cstone::ballotSync(repeat)) { break; }
            if (repeat) { h[i] = updateH(ng0, ncSph, h[i]); }
            ncSph =
                1 + traverseNeighbors(bodyBegin, bodyEnd, x, y, z, h, tree, box, neighborsWarp, ngmax, globalPool)[0];

            if (ncIt == ncMaxIteration) { nc_h_convergenceFailure = true; }
        }

        if (i >= bodyEnd) continue;

        unsigned ncCapped = stl::min(ncSph - 1, ngmax);
        xm[i] = sph::xmassJLoop<TravConfig::targetSize>(i, K, box, neighborsWarp + laneIdx, ncCapped, x, y, z, h, m, wh,
                                                        whd);
        nc[i] = ncSph;
    }
}

template<class Dataset>
void computeXMass(size_t startIndex, size_t endIndex, Dataset& d, const cstone::Box<typename Dataset::RealType>& box)
{
    unsigned numBodies = endIndex - startIndex;
    unsigned numBlocks = TravConfig::numBlocks(numBodies);

    auto [traversalPool, nidxPool] = cstone::allocateNcStacks(d.devData.traversalStack, numBodies, d.ngmax);
    cstone::resetTraversalCounters<<<1, 1>>>();

    unsigned numGroups = d.devData.targetGroups.size() - 1;
    xmassGpu<<<numBlocks, TravConfig::numThreads>>>(
        d.K, d.ng0, d.ngmax, box, rawPtr(d.devData.targetGroups), numGroups, d.treeView.nsView(), rawPtr(d.devData.nc),
        rawPtr(d.devData.x), rawPtr(d.devData.y), rawPtr(d.devData.z), rawPtr(d.devData.h), rawPtr(d.devData.m),
        rawPtr(d.devData.wh), rawPtr(d.devData.whd), rawPtr(d.devData.xm), nidxPool, traversalPool);
    checkGpuErrors(cudaDeviceSynchronize());

    NcStats::type stats[NcStats::numStats];
    checkGpuErrors(cudaMemcpyFromSymbol(stats, cstone::ncStats, NcStats::numStats * sizeof(NcStats::type)));

    bool convergenceFailure;
    checkGpuErrors(cudaMemcpyFromSymbol(&convergenceFailure, nc_h_convergenceFailure, sizeof(bool)));

    NcStats::type maxP2P   = stats[cstone::NcStats::maxP2P];
    NcStats::type maxStack = stats[cstone::NcStats::maxStack];

    d.devData.stackUsedNc = maxStack;

    if (maxP2P == 0xFFFFFFFF) { throw std::runtime_error("GPU traversal stack exhausted in neighbor search\n"); }
    if (convergenceFailure) { throw std::runtime_error("coupled nc/h-updated failed to converge"); }
}

template void computeXMass(size_t, size_t, sphexa::ParticlesData<cstone::GpuTag>& d,
                           const cstone::Box<SphTypes::CoordinateType>&);

template<class Dataset>
void computeDensity(size_t startIndex, size_t endIndex, Dataset& d, const cstone::Box<typename Dataset::RealType>& box)
{
    swap(d.devData.xm, d.devData.rho);
    computeXMass(startIndex, endIndex, d, box);
    swap(d.devData.xm, d.devData.rho);

    // rho[i] = m[i] / rho[i];
    thrust::transform(d.devData.m.begin() + startIndex, d.devData.m.begin() + endIndex,
                      d.devData.rho.begin() + startIndex, d.devData.rho.begin() + startIndex,
                      thrust::divides<typename decltype(d.devData.m)::value_type>{});
}

template void computeDensity(size_t, size_t, sphexa::ParticlesData<cstone::GpuTag>& d,
                             const cstone::Box<SphTypes::CoordinateType>&);

} // namespace cuda

template<class Dataset>
void computeTargetGroups(size_t startIndex, size_t endIndex, Dataset& d,
                         const cstone::Box<typename Dataset::RealType>& box)
{
    thrust::device_vector<util::array<GpuConfig::ThreadMask, TravConfig::nwt>> S;

    float tolFactor = 2.0f;
    cstone::computeGroupSplits<TravConfig::targetSize>(startIndex, endIndex, rawPtr(d.devData.x), rawPtr(d.devData.y),
                                                       rawPtr(d.devData.z), rawPtr(d.devData.h), d.treeView.leaves,
                                                       d.treeView.tree.numLeafNodes, d.treeView.layout, box, tolFactor,
                                                       S, d.devData.traversalStack, d.devData.targetGroups);
}

template void computeTargetGroups(size_t, size_t, sphexa::ParticlesData<cstone::GpuTag>& d,
                                  const cstone::Box<SphTypes::CoordinateType>&);

} // namespace sph
