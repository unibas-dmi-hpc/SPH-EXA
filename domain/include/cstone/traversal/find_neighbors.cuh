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
 * @brief Neighbor search on GPU with breadth-first warp-aware octree traversal
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <algorithm>

#include "cstone/cuda/gpu_config.cuh"
#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/primitives/warpscan.cuh"
#include "cstone/tree/definitions.h"

namespace cstone
{

struct TravConfig
{
    //! @brief size of global workspace memory per warp
    static constexpr int memPerWarp = 2048 * GpuConfig::warpSize;
    //! @brief number of threads per block for the traversal kernel
    static constexpr int numThreads = 256;

    static constexpr int numWarpsPerSm = 20;
    //! @brief maximum number of simultaneously active blocks
    inline static int maxNumActiveBlocks =
        GpuConfig::smCount * (TravConfig::numWarpsPerSm / (TravConfig::numThreads / GpuConfig::warpSize));

    //! @brief number of particles per target, i.e. per warp
    static constexpr int targetSize = 64;

    //! @brief number of warps per target, used all over the place, hence the short name
    static constexpr int nwt = targetSize / GpuConfig::warpSize;
};

__device__ __forceinline__ int ringAddr(const int i) { return i & (TravConfig::memPerWarp - 1); }

/*! @brief count neighbors within a cutoff
 *
 * @tparam       T             float or double
 * @param[in]    sourceBody    source body x,y,z
 * @param[in]    validLaneMask number of lanes that contain valid source bodies
 * @param[in]    pos_i         target body x,y,z,h
 * @param[in]    box           global coordinate bounding box
 * @param[in]    sourceBodyIdx index of source body of each lane
 * @param[in]    ngmax         maximum number of neighbors to store
 * @param[inout] nc_i          target neighbor count to add to
 * @param[out]   nidx_i        indices of neighbors of the target body to store
 *
 * Number of computed particle-particle pairs per call is GpuConfig::warpSize^2 * TravConfig::nwt
 */
template<bool UsePbc, class Tc>
__device__ void countNeighbors(Vec3<Tc> sourceBody,
                               int numLanesValid,
                               const util::array<Vec4<Tc>, TravConfig::nwt>& pos_i,
                               const Box<Tc>& box,
                               cstone::LocalIndex sourceBodyIdx,
                               unsigned ngmax,
                               unsigned nc_i[TravConfig::nwt],
                               unsigned* nidx_i)
{
    const int laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);

    for (int j = 0; j < numLanesValid; j++)
    {
        Vec3<Tc> pos_j{shflSync(sourceBody[0], j), shflSync(sourceBody[1], j), shflSync(sourceBody[2], j)};
        cstone::LocalIndex idx_j = shflSync(sourceBodyIdx, j);

#pragma unroll
        for (int k = 0; k < TravConfig::nwt; k++)
        {
            Tc d2 = distanceSq<UsePbc>(pos_j[0], pos_j[1], pos_j[2], pos_i[k][0], pos_i[k][1], pos_i[k][2], box);
            if (d2 < pos_i[k][3] && d2 > Tc(0.0))
            {
                if (nc_i[k] < ngmax)
                {
                    nidx_i[nc_i[k] * TravConfig::targetSize + laneIdx + k * GpuConfig::warpSize] = idx_j;
                }
                nc_i[k]++;
            }
        }
    }
}

template<bool UsePbc, class T, std::enable_if_t<UsePbc, int> = 0>
__device__ __forceinline__ bool cellOverlap(const Vec3<T>& curSrcCenter,
                                            const Vec3<T>& curSrcSize,
                                            const Vec3<T>& targetCenter,
                                            const Vec3<T>& targetSize,
                                            const Box<T>& box)
{
    return norm2(minDistance(curSrcCenter, curSrcSize, targetCenter, targetSize, box)) == T(0.0);
}

template<bool UsePbc, class T, std::enable_if_t<!UsePbc, int> = 0>
__device__ __forceinline__ bool cellOverlap(const Vec3<T>& curSrcCenter,
                                            const Vec3<T>& curSrcSize,
                                            const Vec3<T>& targetCenter,
                                            const Vec3<T>& targetSize,
                                            const Box<T>& /*box*/)
{
    return norm2(minDistance(curSrcCenter, curSrcSize, targetCenter, targetSize)) == T(0.0);
}

template<class Tc>
__device__ __forceinline__ bool tightOverlap(int laneIdx,
                                             bool isClose,
                                             const Vec3<Tc>& srcCenter,
                                             const Vec3<Tc>& srcSize,
                                             const util::array<Vec4<Tc>, TravConfig::nwt>& pos_i,
                                             const cstone::Box<Tc>& box)
{
    GpuConfig::ThreadMask closeLanes = ballotSync(isClose);

    bool isTightClose = isClose;
    for (unsigned lane = 0 ; lane < GpuConfig::warpSize; ++lane)
    {
        // skip if this lane does not have a close source
        if (!((GpuConfig::ThreadMask(1) << lane) & closeLanes)) { continue; }

        // broadcast srcCenter/size of this lane
        Vec3<Tc> center{shflSync(srcCenter[0], lane), shflSync(srcCenter[1], lane), shflSync(srcCenter[2], lane)};
        Vec3<Tc> size{shflSync(srcSize[0], lane), shflSync(srcSize[1], lane), shflSync(srcSize[2], lane)};

        // does any of the individual target particles overlap with center/size ?
        bool overlapsWithLaneParticle = false;
        for (unsigned k = 0; k < TravConfig::nwt; ++k)
        {
            overlapsWithLaneParticle |= norm2(minDistance(makeVec3(pos_i[k]), center, size, box)) < pos_i[k][3];
        }
        GpuConfig::ThreadMask anyOverlaps = ballotSync(overlapsWithLaneParticle);
        if (lane == laneIdx) { isTightClose = anyOverlaps; }
    }
    return isTightClose;
}

/*! @brief traverse one warp with up to 64 target bodies down the tree
 *
 * @param[inout] acc_i          acceleration of target to add to, TravConfig::nwt Vec4 per lane
 * @param[in]    pos_i          target positions, and smoothing lengths, TravConfig::nwt per lane
 * @param[in]    targetCenter   geometrical target center
 * @param[in]    targetSize     geometrical target bounding box size
 * @param[in]    x,y,z,m,h      source bodies as referenced by tree cells
 * @param[in]    childOffsets   location (index in [0:numTreeNodes]) of first child of each cell, 0 indicates a leaf
 * @param[in]    internalToLeaf for each cell in [0:numTreeNodes], stores the leaf cell (cstone) index in [0:numLeaves]
 *                              if the cell is not a leaf, the value is negative
 * @param[in]    layout         for each leaf cell in [0:numLeaves], stores the index of the first body in the cell
 * @param[in]    sourceCenter   x,y,z center and square MAC radius of each cell in [0:numTreeNodes]
 * @param[in]    Multipoles     the multipole expansions in the same order as srcCells
 * @param[in]    rootRange      source cell indices indices of the top 8 octants
 * @param[-]     tempQueue      shared mem int pointer to GpuConfig::warpSize ints, uninitialized
 * @param[-]     cellQueue      pointer to global memory, 4096 ints per thread, uninitialized
 * @return                      Number of M2P and P2P interactions applied to the group of target particles.
 *                              The totals for the warp are the numbers returned here times the number of valid
 *                              targets in the warp.
 *
 * Constant input pointers are additionally marked __restrict__ to indicate to the compiler that loads
 * can be routed through the read-only/texture cache.
 */
template<bool UsePbc, class Tc, class Th, class KeyType>
__device__ unsigned traverseWarp(unsigned* nc_i,
                                 unsigned* nidx_i,
                                 unsigned ngmax,
                                 const util::array<Vec4<Tc>, TravConfig::nwt>& pos_i,
                                 const Vec3<Tc> targetCenter,
                                 const Vec3<Tc> targetSize,
                                 const Tc* __restrict__ x,
                                 const Tc* __restrict__ y,
                                 const Tc* __restrict__ z,
                                 const Th* __restrict__ /*h*/,
                                 const OctreeNsView<KeyType, Tc>& tree,
                                 int2 rootRange,
                                 const Box<Tc>& box,
                                 volatile int* tempQueue,
                                 int* cellQueue)
{
    const TreeNodeIndex* __restrict__ childOffsets   = tree.childOffsets;
    const TreeNodeIndex* __restrict__ internalToLeaf = tree.internalToLeaf;
    const LocalIndex* __restrict__ layout            = tree.layout;
    const Vec3<Tc>* __restrict__ centers             = tree.centers;
    const Vec3<Tc>* __restrict__ sizes               = tree.sizes;

    const int laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);

    unsigned p2pCounter = 0;

    int bodyQueue; // warp queue for source body indices

    // populate initial cell queue
    for (int root = rootRange.x; root < rootRange.y; root += GpuConfig::warpSize)
    {
        if (root + laneIdx < rootRange.y) { cellQueue[ringAddr(root - rootRange.x + laneIdx)] = root + laneIdx; }
    }

    // these variables are always identical on all warp lanes
    int numSources   = rootRange.y - rootRange.x; // current stack size
    int newSources   = 0;                         // stack size for next level
    int oldSources   = 0;                         // cell indices done
    int sourceOffset = 0; // current level stack pointer, once this reaches numSources, the level is done
    int bdyFillLevel = 0; // fill level of the source body warp queue

    while (numSources > 0) // While there are source cells to traverse
    {
        const int sourceIdx         = sourceOffset + laneIdx;                      // Source cell index of current lane
        int sourceQueue             = cellQueue[ringAddr(oldSources + sourceIdx)]; // Global source cell index in queue
        const Vec3<Tc> curSrcCenter = centers[sourceQueue];                        // Current source cell center
        const Vec3<Tc> curSrcSize   = sizes[sourceQueue];                          // Current source cell center
        const int childBegin        = childOffsets[sourceQueue];                   // First child cell
        const bool isNode           = childBegin;
        const bool isClose          = cellOverlap<UsePbc>(curSrcCenter, curSrcSize, targetCenter, targetSize, box);
        const bool isSource         = sourceIdx < numSources; // Source index is within bounds
        const bool isDirect         = isClose && !isNode && isSource;
        const int leafIdx           = (isDirect) ? internalToLeaf[sourceQueue] : 0; // the cstone leaf index

        // Split
        const bool isSplit     = isNode && isClose && isSource; // Source cell must be split
        const int numChild     = 8 & -int(isSplit);             // Number of child cells (masked by split flag)
        const int numChildScan = inclusiveScanInt(numChild);    // Inclusive scan of numChild
        const int numChildLane = numChildScan - numChild;       // Exclusive scan of numChild
        const int numChildWarp = shflSync(numChildScan, GpuConfig::warpSize - 1); // Total numChild of current warp
        sourceOffset += imin(GpuConfig::warpSize, numSources - sourceOffset);     // advance current level stack pointer
        if (numChildWarp + numSources - sourceOffset > TravConfig::memPerWarp)    // If cell queue overflows
            return 0xFFFFFFFF;                                                    // Exit kernel
        int childIdx = oldSources + numSources + newSources + numChildLane;       // Child index of current lane
        for (int i = 0; i < numChild; i++)                                        // Loop over child cells for each lane
            cellQueue[ringAddr(childIdx + i)] = childBegin + i;                   // Queue child cells for next level
        newSources += numChildWarp; //  Increment source cell count for next loop

        // Direct
        const int firstBody     = layout[leafIdx];
        const int numBodies     = (layout[leafIdx + 1] - firstBody) & -int(isDirect); // Number of bodies in cell
        bool directTodo         = numBodies;
        const int numBodiesScan = inclusiveScanInt(numBodies);                      // Inclusive scan of numBodies
        int numBodiesLane       = numBodiesScan - numBodies;                        // Exclusive scan of numBodies
        int numBodiesWarp       = shflSync(numBodiesScan, GpuConfig::warpSize - 1); // Total numBodies of current warp
        int prevBodyIdx         = 0;
        while (numBodiesWarp > 0) // While there are bodies to process from current source cell set
        {
            tempQueue[laneIdx] = 1; // Default scan input is 1, such that consecutive lanes load consecutive bodies
            if (directTodo && (numBodiesLane < GpuConfig::warpSize))
            {
                directTodo               = false;          // Set cell as processed
                tempQueue[numBodiesLane] = -1 - firstBody; // Put first source cell body index into the queue
            }
            const int bodyIdx = inclusiveSegscanInt(tempQueue[laneIdx], prevBodyIdx);
            // broadcast last processed bodyIdx from the last lane to restart the scan in the next iteration
            prevBodyIdx = shflSync(bodyIdx, GpuConfig::warpSize - 1);

            if (numBodiesWarp >= GpuConfig::warpSize) // Process bodies from current set of source cells
            {
                // Load source body coordinates
                const Vec3<Tc> sourceBody = {x[bodyIdx], y[bodyIdx], z[bodyIdx]};
                countNeighbors<UsePbc>(sourceBody, GpuConfig::warpSize, pos_i, box, bodyIdx, ngmax, nc_i, nidx_i);
                numBodiesWarp -= GpuConfig::warpSize;
                numBodiesLane -= GpuConfig::warpSize;
                p2pCounter += GpuConfig::warpSize;
            }
            else // Fewer than warpSize bodies remaining from current source cell set
            {
                // push the remaining bodies into bodyQueue
                int topUp = shflUpSync(bodyIdx, bdyFillLevel);
                bodyQueue = (laneIdx < bdyFillLevel) ? bodyQueue : topUp;

                bdyFillLevel += numBodiesWarp;
                if (bdyFillLevel >= GpuConfig::warpSize) // If this causes bodyQueue to spill
                {
                    // Load source body coordinates
                    const Vec3<Tc> sourceBody = {x[bodyQueue], y[bodyQueue], z[bodyQueue]};
                    countNeighbors<UsePbc>(sourceBody, GpuConfig::warpSize, pos_i, box, bodyQueue, ngmax, nc_i, nidx_i);
                    bdyFillLevel -= GpuConfig::warpSize;
                    // bodyQueue is now empty; put body indices that spilled into the queue
                    bodyQueue = shflDownSync(bodyIdx, numBodiesWarp - bdyFillLevel);
                    p2pCounter += GpuConfig::warpSize;
                }
                numBodiesWarp = 0; // No more bodies to process from current source cells
            }
        }

        //  If the current level is done
        if (sourceOffset >= numSources)
        {
            oldSources += numSources;      // Update finished source size
            numSources   = newSources;     // Update current source size
            sourceOffset = newSources = 0; // Initialize next source size and offset
        }
    }

    if (bdyFillLevel > 0) // If there are leftover direct bodies
    {
        const bool laneHasBody = laneIdx < bdyFillLevel;
        // Load position of source bodies, with padding for invalid lanes
        const Vec3<Tc> sourceBody =
            laneHasBody ? Vec3<Tc>{x[bodyQueue], y[bodyQueue], z[bodyQueue]} : Vec3<Tc>{Tc(0), Tc(0), Tc(0)};
        countNeighbors<UsePbc>(sourceBody, bdyFillLevel, pos_i, box, bodyQueue, ngmax, nc_i, nidx_i);
        p2pCounter += bdyFillLevel;
    }

    return p2pCounter;
}

static __device__ unsigned long long sumNcP2PGlob = 0;
static __device__ unsigned maxNcP2PGlob           = 0;
static __device__ unsigned targetCounterGlob      = 0;

static __global__ void resetTraversalCounters()
{
    sumNcP2PGlob = 0;
    maxNcP2PGlob = 0;
    targetCounterGlob = 0;
}

template<class Tc, class Th, class Index>
__device__ __forceinline__ util::array<Vec4<Tc>, TravConfig::nwt> loadTarget(Index bodyBegin,
                                                                             Index bodyEnd,
                                                                             unsigned lane,
                                                                             const Tc* __restrict__ x,
                                                                             const Tc* __restrict__ y,
                                                                             const Tc* __restrict__ z,
                                                                             const Th* __restrict__ h)
{
    util::array<Vec4<Tc>, TravConfig::nwt> pos_i;
#pragma unroll
    for (int i = 0; i < TravConfig::nwt; i++)
    {
        Index bodyIdx = imin(bodyBegin + i * GpuConfig::warpSize + lane, bodyEnd - 1);
        pos_i[i]      = {x[bodyIdx], y[bodyIdx], z[bodyIdx], Tc(2) * h[bodyIdx]};
    }
    return pos_i;
}

//! @brief determine the bounding box around all particles-2h spheres in the warp
template<class Tc>
__device__ __forceinline__ thrust::tuple<Vec3<Tc>, Vec3<Tc>>
warpBbox(const util::array<Vec4<Tc>, TravConfig::nwt>& pos_i)
{
    Tc r0 = pos_i[0][3];
    Vec3<Tc> Xmin{pos_i[0][0] - r0, pos_i[0][1] - r0, pos_i[0][2] - r0};
    Vec3<Tc> Xmax{pos_i[0][0] + r0, pos_i[0][1] + r0, pos_i[0][2] + r0};
#pragma unroll
    for (int i = 1; i < TravConfig::nwt; i++)
    {
        Tc ri = pos_i[i][3];
        Vec3<Tc> iboxMin{pos_i[i][0] - ri, pos_i[i][1] - ri, pos_i[i][2] - ri};
        Vec3<Tc> iboxMax{pos_i[i][0] + ri, pos_i[i][1] + ri, pos_i[i][2] + ri};
        Xmin = min(Xmin, iboxMin);
        Xmax = max(Xmax, iboxMax);
    }

    Xmin = {warpMin(Xmin[0]), warpMin(Xmin[1]), warpMin(Xmin[2])};
    Xmax = {warpMax(Xmax[0]), warpMax(Xmax[1]), warpMax(Xmax[2])};

    const Vec3<Tc> targetCenter = (Xmax + Xmin) * Tc(0.5);
    const Vec3<Tc> targetSize   = (Xmax - Xmin) * Tc(0.5);

    return thrust::make_tuple(targetCenter, targetSize);
}

template<class Tc, class Th, class KeyType>
__device__ util::array<unsigned, 2> traverseNeighbors(cstone::LocalIndex bodyBegin,
                                                      cstone::LocalIndex bodyEnd,
                                                      const int2 rootRange,
                                                      const Tc* __restrict__ x,
                                                      const Tc* __restrict__ y,
                                                      const Tc* __restrict__ z,
                                                      const Th* __restrict__ h,
                                                      const OctreeNsView<KeyType, Tc>& tree,
                                                      const Box<Tc> box,
                                                      unsigned* warpNidx,
                                                      unsigned ngmax,
                                                      int* globalPool)
{
    const unsigned laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);
    const unsigned warpIdx = threadIdx.x >> GpuConfig::warpSizeLog2;

    constexpr unsigned numWarpsPerBlock = TravConfig::numThreads / GpuConfig::warpSize;

    __shared__ int sharedPool[TravConfig::numThreads];

    // warp-common shared mem, 1 int per thread
    int* tempQueue = sharedPool + GpuConfig::warpSize * warpIdx;
    // warp-common global mem storage
    int* cellQueue = globalPool + TravConfig::memPerWarp * ((blockIdx.x * numWarpsPerBlock) + warpIdx);

    util::array<Vec4<Tc>, TravConfig::nwt> pos_i = loadTarget(bodyBegin, bodyEnd, laneIdx, x, y, z, h);
    const auto [targetCenter, targetSize] = warpBbox(pos_i);

#pragma unroll
    for (int k = 0; k < TravConfig::nwt; ++k)
    {
        auto r      = pos_i[k][3];
        pos_i[k][3] = r * r;
    }

    auto pbc    = BoundaryType::periodic;
    bool anyPbc = box.boundaryX() == pbc || box.boundaryY() == pbc || box.boundaryZ() == pbc;
    bool usePbc = anyPbc && !insideBox(targetCenter, targetSize, box);

    util::array<unsigned, 2> nc_i;
    nc_i *= 0u;

    unsigned numP2P;
    if (usePbc)
    {
        numP2P = traverseWarp<true>(nc_i.data(), warpNidx, ngmax, pos_i, targetCenter, targetSize, x, y, z, h, tree,
                                    rootRange, box, tempQueue, cellQueue);
    }
    else
    {
        numP2P = traverseWarp<false>(nc_i.data(), warpNidx, ngmax, pos_i, targetCenter, targetSize, x, y, z, h, tree,
                                     rootRange, box, tempQueue, cellQueue);
    }
    assert(numP2P != 0xFFFFFFFF);

    if (laneIdx == 0)
    {
        unsigned targetGroupSize = bodyEnd - bodyBegin;
        atomicMax(&maxNcP2PGlob, numP2P);
        atomicAdd(&sumNcP2PGlob, numP2P * targetGroupSize);
    }

    return nc_i;
}

/*! @brief Neighbor search for bodies within the specified range
 *
 * @param[in]    firstBody           index of first body in @p bodyPos to compute acceleration for
 * @param[in]    lastBody            index (exclusive) of last body in @p bodyPos to compute acceleration for
 * @param[in]    rootRange           (start,end) index pair of cell indices to start traversal from
 * @param[in]    x,y,z,h             bodies, in SFC order and as referenced by @p layout
 * @param[in]    tree.childOffsets   location (index in [0:numTreeNodes]) of first child of each cell, 0 indicates a
 *                                   leaf
 * @param[in]    tree.internalToLeaf for each cell in [0:numTreeNodes], stores the leaf cell (cstone) index in
 *                                   [0:numLeaves] if the cell is not a leaf, the value is negative
 * @param[in]    tree.layout         for each leaf cell in [0:numLeaves], stores the index of the first body in the cell
 * @param[in]    tree.centers        x,y,z geometric center of each cell in [0:numTreeNodes]
 * @param[in]    tree.sizes          x,y,z geometric size of each cell in [0:numTreeNodes]
 * @param[in]    box                 global coordinate bounding box
 * @param[out]   nc                  neighbor counts of bodies with indices in [firstBody, lastBody]
 * @param[-]     globalPool          temporary storage for the cell traversal stack, uninitialized
 *                                   each active warp needs space for TravConfig::memPerWarp int32,
 *                                   so the total size is TravConfig::memPerWarp * numWarpsPerBlock * numBlocks
 */
template<class Tc, class Th, class KeyType>
__global__ __launch_bounds__(TravConfig::numThreads) void traverseBT(cstone::LocalIndex firstBody,
                                                                     cstone::LocalIndex lastBody,
                                                                     const int2 rootRange,
                                                                     const Tc* __restrict__ x,
                                                                     const Tc* __restrict__ y,
                                                                     const Tc* __restrict__ z,
                                                                     const Th* __restrict__ h,
                                                                     OctreeNsView<KeyType, Tc> tree,
                                                                     const Box<Tc> box,
                                                                     unsigned* nc,
                                                                     unsigned* nidx,
                                                                     unsigned ngmax,
                                                                     int* globalPool)
{
    const unsigned laneIdx    = threadIdx.x & (GpuConfig::warpSize - 1);
    const unsigned numTargets = (lastBody - firstBody - 1) / TravConfig::targetSize + 1;
    int targetIdx             = 0;

    while (true)
    {
        // first thread in warp grabs next target
        if (laneIdx == 0) { targetIdx = atomicAdd(&targetCounterGlob, 1); }
        targetIdx = shflSync(targetIdx, 0);

        if (targetIdx >= numTargets) return;

        const cstone::LocalIndex bodyBegin = firstBody + targetIdx * TravConfig::targetSize;
        const cstone::LocalIndex bodyEnd   = imin(bodyBegin + TravConfig::targetSize, lastBody);
        unsigned* warpNidx                 = nidx + targetIdx * TravConfig::targetSize * ngmax;

        auto nc_i =
            traverseNeighbors(bodyBegin, bodyEnd, rootRange, x, y, z, h, tree, box, warpNidx, ngmax, globalPool);

        const cstone::LocalIndex bodyIdxLane = bodyBegin + laneIdx;
        for (int i = 0; i < TravConfig::nwt; i++)
        {
            const cstone::LocalIndex bodyIdx = bodyIdxLane + i * GpuConfig::warpSize;
            if (bodyIdx < bodyEnd) { nc[bodyIdx] = nc_i[i]; }
        }
    }
}

/*! @brief Compute approximate body accelerations with Barnes-Hut
 *
 * @param[in]    firstBody      index of first body in @p bodyPos to compute acceleration for
 * @param[in]    lastBody       index (exclusive) of last body in @p bodyPos to compute acceleration for
 * @param[in]    x,y,z,h        bodies, in SFC order and as referenced by @p layout
 * @param[in]    childOffsets   location (index in [0:numTreeNodes]) of first child of each cell, 0 indicates a leaf
 * @param[in]    internalToLeaf for each cell in [0:numTreeNodes], stores the leaf cell (cstone) index in [0:numLeaves]
 *                              if the cell is not a leaf, the value is negative
 * @param[in]    layout         for each leaf cell in [0:numLeaves], stores the index of the first body in the cell
 * @param[in]    box            global coordinate bounding box
 * @param[out]   nc             output neighbor counts
 * @return                      interaction statistics
 */
template<class Tc, class Th, class KeyType>
auto findNeighborsBT(size_t firstBody,
                     size_t lastBody,
                     const Tc* x,
                     const Tc* y,
                     const Tc* z,
                     const Th* h,
                     OctreeNsView<KeyType, Tc> tree,
                     const Box<Tc>& box,
                     unsigned* nc,
                     unsigned* nidx,
                     unsigned ngmax)
{
    constexpr int numWarpsPerBlock = TravConfig::numThreads / GpuConfig::warpSize;

    int numBodies = lastBody - firstBody;

    // each target gets a warp (numWarps == numTargets)
    int numWarps  = (numBodies - 1) / TravConfig::targetSize + 1;
    int numBlocks = (numWarps - 1) / numWarpsPerBlock + 1;
    numBlocks     = std::min(numBlocks, TravConfig::maxNumActiveBlocks);

    printf("launching %d blocks\n", numBlocks);

    const int poolSize = TravConfig::memPerWarp * numWarpsPerBlock * numBlocks;
    thrust::device_vector<int> globalPool(poolSize);

    resetTraversalCounters<<<1, 1>>>();
    auto t0 = std::chrono::high_resolution_clock::now();
    traverseBT<<<numBlocks, TravConfig::numThreads>>>(firstBody, lastBody, {1, 9}, x, y, z, h, tree, box, nc, nidx,
                                                      ngmax, rawPtr(globalPool));
    kernelSuccess("traverseBT");

    auto t1   = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();

    uint64_t sumP2P;
    unsigned maxP2P;

    checkGpuErrors(cudaMemcpyFromSymbol(&sumP2P, sumNcP2PGlob, sizeof(uint64_t)));
    checkGpuErrors(cudaMemcpyFromSymbol(&maxP2P, maxNcP2PGlob, sizeof(unsigned int)));

    util::array<Tc, 2> interactions;
    interactions[0] = Tc(sumP2P) / Tc(numBodies);
    interactions[1] = Tc(maxP2P);

    fprintf(stdout, "Traverse : %.7f s (%.7f TFlops) P2P %f, maxP2P %f\n", dt, 11.0 * sumP2P / dt / 1e12,
            interactions[0], interactions[1]);

    return interactions;
}

} // namespace cstone
