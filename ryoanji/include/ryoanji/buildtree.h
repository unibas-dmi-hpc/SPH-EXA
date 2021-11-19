#pragma once

#include <chrono>

#include "cudavec.h"
#include "types.h"
#include "warpscan.h"

extern void sort(const int size, int* key, int* value);

namespace
{
__constant__ int maxNodesGlob;
__device__ Box boxGlob;
__device__ unsigned int counterGlob   = 0;
__device__ unsigned int numNodesGlob  = 0;
__device__ unsigned int numLeafsGlob  = 0;
__device__ unsigned int numLevelsGlob = 0;
__device__ unsigned int numCellsGlob  = 0;
__device__ int* octantSizePool;
__device__ int* octantSizeScanPool;
__device__ int* subOctantSizeScanPool;
__device__ int* blockCounterPool;
__device__ int2* bodyRangePool;
__device__ CellData* sourceCells;

__device__ __forceinline__ int getOctant(const Box& box, const fvec4& body)
{
    return ((box.X[0] <= body[0]) << 0) + ((box.X[1] <= body[1]) << 1) + ((box.X[2] <= body[2]) << 2);
}

__device__ __forceinline__ Box getChild(const Box& box, const int octant)
{
    const float R = 0.5f * box.R;
    const fvec3 X(box.X[0] + R * (octant & 1 ? 1.0f : -1.0f),
                  box.X[1] + R * (octant & 2 ? 1.0f : -1.0f),
                  box.X[2] + R * (octant & 4 ? 1.0f : -1.0f));

    Box childBox = {X, R};

    return childBox;
}

__device__ fvec3 minBlock(fvec3 Xmin)
{
    const int laneIdx = threadIdx.x & (WARP_SIZE - 1);
    const int warpIdx = threadIdx.x >> WARP_SIZE2;

    #pragma unroll
    for (int i = 0; i < WARP_SIZE2; i++)
    {
        const int offset = 1 << i;

        Xmin[0] = min(Xmin[0], __shfl_xor_sync(0xFFFFFFFF, Xmin[0], offset));
        Xmin[1] = min(Xmin[1], __shfl_xor_sync(0xFFFFFFFF, Xmin[1], offset));
        Xmin[2] = min(Xmin[2], __shfl_xor_sync(0xFFFFFFFF, Xmin[2], offset));
    }

    const int NWARP2 = NTHREAD2 - WARP_SIZE2;
    const int NWARP  = 1 << NWARP2;

    __shared__ float3 shared[NWARP];
    fvec3* sharedXmin = reinterpret_cast<fvec3*>(shared);
    if (laneIdx == 0) { sharedXmin[warpIdx] = Xmin; }

    __syncthreads();

    if (threadIdx.x < NWARP)
    {
        Xmin = sharedXmin[threadIdx.x];
        #pragma unroll
        for (int i = 0; i < NWARP2; i++)
        {
            const int offset = 1 << i;

            Xmin[0] = min(Xmin[0], __shfl_xor_sync(0xFFFFFFFF, Xmin[0], offset));
            Xmin[1] = min(Xmin[1], __shfl_xor_sync(0xFFFFFFFF, Xmin[1], offset));
            Xmin[2] = min(Xmin[2], __shfl_xor_sync(0xFFFFFFFF, Xmin[2], offset));
        }
    }

    return Xmin;
}

__device__ fvec3 maxBlock(fvec3 Xmax)
{
    const int laneIdx = threadIdx.x & (WARP_SIZE - 1);
    const int warpIdx = threadIdx.x >> WARP_SIZE2;

    #pragma unroll
    for (int i = 0; i < WARP_SIZE2; i++)
    {
        const int offset = 1 << i;

        Xmax[0] = max(Xmax[0], __shfl_xor_sync(0xFFFFFFFF, Xmax[0], offset));
        Xmax[1] = max(Xmax[1], __shfl_xor_sync(0xFFFFFFFF, Xmax[1], offset));
        Xmax[2] = max(Xmax[2], __shfl_xor_sync(0xFFFFFFFF, Xmax[2], offset));
    }

    const int NWARP2 = NTHREAD2 - WARP_SIZE2;
    const int NWARP  = 1 << NWARP2;

    __shared__ float3 shared[NWARP];
    fvec3* sharedXmax = reinterpret_cast<fvec3*>(shared);
    if (laneIdx == 0) { sharedXmax[warpIdx] = Xmax; }

    __syncthreads();

    if (threadIdx.x < NWARP)
    {
        Xmax = sharedXmax[threadIdx.x];
        #pragma unroll
        for (int i = 0; i < NWARP2; i++)
        {
            const int offset = 1 << i;

            Xmax[0] = max(Xmax[0], __shfl_xor_sync(0xFFFFFFFF, Xmax[0], offset));
            Xmax[1] = max(Xmax[1], __shfl_xor_sync(0xFFFFFFFF, Xmax[1], offset));
            Xmax[2] = max(Xmax[2], __shfl_xor_sync(0xFFFFFFFF, Xmax[2], offset));
        }
    }

    return Xmax;
}

/*! @brief compute coordinate bounds of given bodies
 *
 * @param[in]  numBodies
 * @param[out] bounds     array of length 2 * NTHREADS, final result returned in element 0
 * @param[in]  bodyPos
 *
 * performs a (min/max)-reduction
 */
__global__ void getBounds(const int numBodies, Bounds* bounds, const fvec4* bodyPos)
{
    const int NBLOCK = NTHREAD;
    const int begin  = blockIdx.x * blockDim.x + threadIdx.x;

    fvec3 Xmin       = INFINITY;
    fvec3 Xmax       = -INFINITY;

    for (int i = begin; i < numBodies; i += NBLOCK * NTHREAD)
    {
        const fvec3 pos = make_fvec3(bodyPos[i]);
        Xmin            = min(Xmin, pos);
        Xmax            = max(Xmax, pos);
    }

    Xmin = minBlock(Xmin);
    Xmax = maxBlock(Xmax);

    if (threadIdx.x == 0)
    {
        bounds[blockIdx.x].Xmin = Xmin;
        bounds[blockIdx.x].Xmax = Xmax;
    }

    __shared__ bool lastBlock;
    __threadfence();
    __syncthreads();

    if (threadIdx.x == 0)
    {
        const int blockCount = atomicInc(&counterGlob, NBLOCK);
        lastBlock            = (blockCount == NBLOCK - 1);
    }

    __syncthreads();

    if (lastBlock)
    {
        Xmin = bounds[threadIdx.x].Xmin;
        Xmax = bounds[threadIdx.x].Xmax;
        Xmin = minBlock(Xmin);
        Xmax = maxBlock(Xmax);

        __syncthreads();

        if (threadIdx.x == 0)
        {
            const fvec3 X = (Xmax + Xmin) * 0.5f;
            const fvec3 R = (Xmax - Xmin) * 0.5f;
            const float r = max(R) * 1.1f;
            const Box box = {X, r};
            boxGlob       = box;
            counterGlob   = 0;
        }
    }
}

/*! @brief recurse down octree level by level
 *
 * @tparam      NCRIT
 * @tparam      ISROOT
 * @param[in]   box4                   global coordinate bounding box
 * @param[in]   cellParentIndex        (0 on first call)
 * @param[in]   cellIndexBase          (0 on first call)
 * @param[in]   packedOctant           (0 on first call)
 * @param[in]   octantSizeBase         body octant counts, length 8
 * @param[in]   octantSizeScanBase     scanned body counts, length 8
 * @param[out]  subOctantSizeScanBase  length 64
 * @param[in]   blockCounterBase       length 1
 * @param[in]   bodyRangeBase          length 1
 * @param[in]   bodyPos
 * @param[out]  bodyPos2
 * @param[in]   level
 */
template<int NCRIT, bool ISROOT>
__global__ /*__launch_bounds__(NTHREAD, 8)*/
void buildOctant(float4 box4, const int cellParentIndex, const int cellIndexBase,
                 const int packedOctant, int* octantSizeBase, int* octantSizeScanBase,
                 int* subOctantSizeScanBase, int* blockCounterBase, int2* bodyRangeBase,
                 float4* bodyPos, float4* bodyPos2, int level = 0)
{
    const int NWARP2       = NTHREAD2 - WARP_SIZE2;
    const int NWARP        = 1 << NWARP2;
    const int laneIdx      = threadIdx.x & (WARP_SIZE - 1);
    const int warpIdx      = threadIdx.x >> WARP_SIZE2;
    int* octantSize        = octantSizeBase + blockIdx.y * 8;
    int* octantSizeScan    = octantSizeScanBase + blockIdx.y * 8;
    int* subOctantSizeScan = subOctantSizeScanBase + blockIdx.y * 64;
    int* blockCounter      = blockCounterBase + blockIdx.y;
    int2* bodyRange        = bodyRangeBase + blockIdx.y;
    const int bodyBegin    = bodyRange->x + blockIdx.x * blockDim.x + warpIdx * WARP_SIZE;
    const int bodyEnd      = bodyRange->y;
    const int numBodies    = bodyRange->y - bodyRange->x;
    const int childOctant  = (packedOctant >> (3 * blockIdx.y)) & 0x7;
    __shared__ int subOctantSizeLane[NWARP * 8 * 8];
    __shared__ int subOctantSize[8 * 8];
    Box box       = {fvec3(box4.x, box4.y, box4.z), box4.w};
    Box* childBox = (Box*)subOctantSize;

    for (int i = 0; i < 8 * 8 * NWARP; i += blockDim.x)
        if (i + threadIdx.x < 8 * 8 * NWARP) subOctantSizeLane[i + threadIdx.x] = 0;

    if (!ISROOT) { box = getChild(box, childOctant); }
    if (laneIdx == 0) { childBox[warpIdx] = getChild(box, warpIdx); } // One child per warp

    __syncthreads();

    for (int i = bodyBegin; i < bodyEnd; i += gridDim.x * blockDim.x)
    {
        const int bodyIdx = min(i + laneIdx, bodyEnd - 1);
        fvec4 pos         = bodyPos[bodyIdx];
        int bodyOctant    = getOctant(box, pos);
        int bodySubOctant = getOctant(childBox[bodyOctant], pos);

        if (i + laneIdx > bodyIdx) { bodyOctant = bodySubOctant = 8; } // Out of bounds lanes

        int octantSizeScanLane = 0;

        #pragma unroll
        for (int octant = 0; octant < 8; octant++)
        {
            const int sumLane = reduceBool(bodyOctant == octant); // Count current octant in warp
            if (octant == laneIdx) octantSizeScanLane = sumLane;  // Use lanes 0-7 for each octant
        }

        int octantOffset;
        if (laneIdx < 8) octantOffset = atomicAdd(&octantSizeScan[laneIdx], octantSizeScanLane); // Global scan

        int bodyIdx2 = -1;

        #pragma unroll
        for (int octant = 0; octant < 8; octant++)
        {
            const int sumLane = reduceBool(bodyOctant == octant);
            if (sumLane > 0)
            {                                                                     // Avoid redundant instructions
                const int index  = exclusiveScanBool(bodyOctant == octant);       // Sparse lane index
                const int offset = __shfl_sync(0xFFFFFFFF, octantOffset, octant); // Global offset
                if (bodyOctant == octant)                                         // Prevent overwrite
                    bodyIdx2 = offset + index;                                    // Sorted index
            }
        }

        if (bodyIdx2 >= 0)
            bodyPos2[bodyIdx2] = make_float4(pos[0], pos[1], pos[2], pos[3]); // Assign value to sort buffer

        int remainder = 32;

        #pragma unroll
        for (int octant = 0; octant < 8; octant++)
        {
            if (remainder == 0) break;
            const int sumLane = reduceBool(bodyOctant == octant);
            if (sumLane > 0)
            {
                const int bodySubOctantValid = bodyOctant == octant ? bodySubOctant : 8;
                #pragma unroll
                for (int subOctant = 0; subOctant < 8; subOctant += 4)
                {
                    const int4 sum4 = make_int4(reduceBool(subOctant + 0 == bodySubOctantValid),
                                                reduceBool(subOctant + 1 == bodySubOctantValid),
                                                reduceBool(subOctant + 2 == bodySubOctantValid),
                                                reduceBool(subOctant + 3 == bodySubOctantValid));
                    if (laneIdx == 0)
                    {
                        int4 subOctantTemp = *(int4*)&subOctantSizeLane[warpIdx * 64 + octant * 8 + subOctant];
                        subOctantTemp.x += sum4.x;
                        subOctantTemp.y += sum4.y;
                        subOctantTemp.z += sum4.z;
                        subOctantTemp.w += sum4.w;
                        *(int4*)&subOctantSizeLane[warpIdx * 64 + octant * 8 + subOctant] = subOctantTemp;
                    }
                }
                remainder -= sumLane;
            }
        }
    }
    __syncthreads(); // Sync subOctantSizeLane

    #pragma unroll
    for (int k = 0; k < 8; k += 4)
    {
        int4 subOctantTemp =
            laneIdx < NWARP ? (*(int4*)&subOctantSizeLane[laneIdx * 64 + warpIdx * 8 + k]) : make_int4(0, 0, 0, 0);

        #pragma unroll
        for (int i = NWARP2 - 1; i >= 0; i--)
        {
            subOctantTemp.x += __shfl_xor_sync(0xFFFFFFFF, subOctantTemp.x, 1 << i, NWARP);
            subOctantTemp.y += __shfl_xor_sync(0xFFFFFFFF, subOctantTemp.y, 1 << i, NWARP);
            subOctantTemp.z += __shfl_xor_sync(0xFFFFFFFF, subOctantTemp.z, 1 << i, NWARP);
            subOctantTemp.w += __shfl_xor_sync(0xFFFFFFFF, subOctantTemp.w, 1 << i, NWARP);
        }
        if (laneIdx == 0) *(int4*)&subOctantSize[warpIdx * 8 + k] = subOctantTemp;
    }

    if (laneIdx < 8)
        if (subOctantSize[warpIdx * 8 + laneIdx] > 0)
            atomicAdd(&subOctantSizeScan[warpIdx * 8 + laneIdx], subOctantSize[warpIdx * 8 + laneIdx]);

    __syncthreads(); // Sync subOctantSizeScan, subOctantSize

    __shared__ bool lastBlock;

    if (threadIdx.x == 0)
    {
        const int blockCount = atomicAdd(blockCounter, 1);
        lastBlock            = (blockCount == gridDim.x - 1);
    }

    __syncthreads(); // Sync lastBlock

    if (!lastBlock) return;

    __syncthreads(); // Sync return

    if (threadIdx.x == 0) atomicCAS(&numLevelsGlob, level, level + 1);

    __syncthreads(); // Sync numLevelsGlob

    const int numBodiesOctant     = octantSize[warpIdx];
    const int bodyEndOctant       = octantSizeScan[warpIdx];
    const int bodyBeginOctant     = bodyEndOctant - numBodiesOctant;
    const int numBodiesOctantLane = laneIdx < 8 ? octantSize[laneIdx] : 0;
    const int numNodesLane        = exclusiveScanBool(numBodiesOctantLane > NCRIT);
    const int numLeafsLane        = exclusiveScanBool(0 < numBodiesOctantLane && numBodiesOctantLane <= NCRIT);
    int* numNodes                 = subOctantSize; // Reuse shared memory
    int* numLeafs                 = subOctantSize + 8;
    int& numNodesScan             = subOctantSize[16];
    int& numCellsScan             = subOctantSize[17];

    if (warpIdx == 0 && laneIdx < 8)
    {
        numNodes[laneIdx] = numNodesLane;
        numLeafs[laneIdx] = numLeafsLane;
    }

    int maxBodiesOctant = numBodiesOctantLane;

    #pragma unroll
    for (int i = 2; i >= 0; i--)
        maxBodiesOctant = max(maxBodiesOctant, __shfl_xor_sync(0xFFFFFFFF, maxBodiesOctant, 1 << i));

    const int numNodesWarp = reduceBool(numBodiesOctantLane > NCRIT);

    if (threadIdx.x == 0 && numNodesWarp > 0)
    {
        numNodesScan = atomicAdd(&numNodesGlob, numNodesWarp);
        assert(numNodesScan < maxNodesGlob);
    }

    const int numChildWarp = reduceBool(numBodiesOctantLane > 0);

    if (threadIdx.x == 0 && numChildWarp > 0)
    {
        numCellsScan = atomicAdd(&numCellsGlob, numChildWarp);
        const CellData cellData(level, cellParentIndex, bodyRange->x, numBodies, numCellsScan, numChildWarp);
        sourceCells[cellIndexBase + blockIdx.y] = cellData;
    }

    __syncthreads(); // Sync numCellsScan, sourceCells

    octantSizeBase        = octantSizePool + numNodesScan * 8; // Global offset
    octantSizeScanBase    = octantSizeScanPool + numNodesScan * 8;
    subOctantSizeScanBase = subOctantSizeScanPool + numNodesScan * 64;
    blockCounterBase      = blockCounterPool + numNodesScan;
    bodyRangeBase         = bodyRangePool + numNodesScan;

    const int nodeOffset = numNodes[warpIdx];
    const int leafOffset = numLeafs[warpIdx];

    if (numBodiesOctant > NCRIT)
    {
        octantSize     = octantSizeBase + nodeOffset * 8; // Warp offset
        octantSizeScan = octantSizeScanBase + nodeOffset * 8;
        blockCounter   = blockCounterBase + nodeOffset;
        bodyRange      = bodyRangeBase + nodeOffset;

        const int newOctantSize = laneIdx < 8 ? subOctantSizeScan[warpIdx * 8 + laneIdx] : 0;
        int newOctantSizeScan   = inclusiveScanInt(newOctantSize);
        newOctantSizeScan -= newOctantSize;
        if (laneIdx < 8)
        {
            octantSizeScan[laneIdx] = bodyBeginOctant + newOctantSizeScan;
            octantSize[laneIdx]     = newOctantSize;
        }
        if (laneIdx == 0)
        {
            *blockCounter = 0;
            bodyRange->x  = bodyBeginOctant;
            bodyRange->y  = bodyEndOctant;
        }
    }

    if (numNodesWarp > 0 && warpIdx == 0)
    {
        int packedOctant = numBodiesOctantLane > NCRIT ? laneIdx << (3 * numNodesLane) : 0;
        #pragma unroll
        for (int i = 4; i >= 0; i--)
            packedOctant |= __shfl_xor_sync(0xFFFFFFFF, packedOctant, 1 << i);

        if (threadIdx.x == 0)
        {
            dim3 NBLOCK = min(max(maxBodiesOctant / NTHREAD, 1), 512);
            NBLOCK.y    = numNodesWarp;
            float4 box4 = make_float4(box.X[0], box.X[1], box.X[2], box.R);
            buildOctant<NCRIT, false><<<NBLOCK, NTHREAD>>>(box4,
                                                           cellIndexBase + blockIdx.y,
                                                           numCellsScan,
                                                           packedOctant,
                                                           octantSizeBase,
                                                           octantSizeScanBase,
                                                           subOctantSizeScanBase,
                                                           blockCounterBase,
                                                           bodyRangeBase,
                                                           bodyPos2,
                                                           bodyPos,
                                                           level + 1);
        }
    }

    if (numBodiesOctant <= NCRIT && numBodiesOctant > 0)
    {
        if (laneIdx == 0)
        {
            atomicAdd(&numLeafsGlob, 1);
            const CellData leafData(
                level + 1, cellIndexBase + blockIdx.y, bodyBeginOctant, bodyEndOctant - bodyBeginOctant);
            sourceCells[numCellsScan + numNodesWarp + leafOffset] = leafData;
        }
        if (!(level & 1)) // if tree depth is even
        {
            for (int bodyIdx = bodyBeginOctant + laneIdx; bodyIdx < bodyEndOctant; bodyIdx += WARP_SIZE)
            {
                if (bodyIdx < bodyEndOctant) { bodyPos[bodyIdx] = bodyPos2[bodyIdx]; }
            }
        }
    }
}

/*! @brief calculate particle count of the 8 children of the root octant
 *
 * @param[in]  numBodies   length of @p bodyPos
 * @param[out] octantSize  array of length 8 * maxNode, first 8 elements are set
 * @param[in]  bodyPos     body positions
 */
__global__ void getRootOctantSize(const int numBodies, int* octantSize, const fvec4* bodyPos)
{
    const int laneIdx     = threadIdx.x & (WARP_SIZE - 1);
    const int begin       = blockIdx.x * blockDim.x + threadIdx.x;
    int octantSizeLane[8] = {0};
    for (int i = begin; i < numBodies; i += gridDim.x * blockDim.x)
    {
        const fvec4 pos  = bodyPos[i];
        const int octant = getOctant(boxGlob, pos);
        octantSizeLane[0] += (octant == 0);
        octantSizeLane[1] += (octant == 1);
        octantSizeLane[2] += (octant == 2);
        octantSizeLane[3] += (octant == 3);
        octantSizeLane[4] += (octant == 4);
        octantSizeLane[5] += (octant == 5);
        octantSizeLane[6] += (octant == 6);
        octantSizeLane[7] += (octant == 7);
    }
    #pragma unroll
    for (int k = 0; k < 8; k++)
    {
        int octantSizeTemp = octantSizeLane[k];
        #pragma unroll
        for (int i = 4; i >= 0; i--)
            octantSizeTemp += __shfl_xor_sync(0xFFFFFFFF, octantSizeTemp, 1 << i);
        if (laneIdx == 0) atomicAdd(&octantSize[k], octantSizeTemp);
    }
}

/*! @brief
 *
 * @tparam       NCRIT
 * @param[in]    numBodies
 * @param[in]    d_sourceCells
 * @param[inout] d_octantSizePool
 * @param[out]   d_octantSizeScanPool
 * @param[out]   d_subOctantSizeScanPool
 * @param[out]   d_blockCounterPool
 * @param[out]   d_bodyRangePool
 * @param[in]    d_bodyPos
 * @param[out]   d_bodyPos2
 */
template<int NCRIT>
__global__ void buildOctree(const int numBodies, CellData* d_sourceCells, int* d_octantSizePool,
                            int* d_octantSizeScanPool, int* d_subOctantSizeScanPool, int* d_blockCounterPool,
                            int2* d_bodyRangePool, float4* d_bodyPos, float4* d_bodyPos2)
{
    sourceCells           = d_sourceCells;
    octantSizePool        = d_octantSizePool;
    octantSizeScanPool    = d_octantSizeScanPool;
    subOctantSizeScanPool = d_subOctantSizeScanPool;
    blockCounterPool      = d_blockCounterPool;
    bodyRangePool         = d_bodyRangePool;

    numNodesGlob  = 0;
    numLeafsGlob  = 0;
    numLevelsGlob = 0;
    numCellsGlob  = 0;

    //! root octant counts
    int* octantSize = new int[8];
    for (int k = 0; k < 8; k++)
        octantSize[k] = octantSizePool[k];

    //! scan of root octant counts
    int* octantSizeScan = new int[8];
    for (int k = 0; k < 8; k++)
        octantSizeScan[k] = k == 0 ? 0 : octantSizeScan[k - 1] + octantSize[k - 1];

    int* subOctantSizeScan = new int[64];
    for (int k = 0; k < 64; k++)
        subOctantSizeScan[k] = 0;

    int* blockCounter = new int;
    *blockCounter     = 0;
    int2* bodyRange   = new int2;
    bodyRange->x      = 0;
    bodyRange->y      = numBodies;
    const int NBLOCK  = min((numBodies - 1) / NTHREAD + 1, 512);
    float4 box4       = make_float4(boxGlob.X[0], boxGlob.X[1], boxGlob.X[2], boxGlob.R);
    buildOctant<NCRIT, true><<<NBLOCK, NTHREAD>>>(
        box4, 0, 0, 0, octantSize, octantSizeScan, subOctantSizeScan, blockCounter, bodyRange, d_bodyPos, d_bodyPos2);

    assert(cudaDeviceSynchronize() == cudaSuccess);

    delete[] octantSize;
    delete[] octantSizeScan;
    delete[] subOctantSizeScan;
    delete blockCounter;
    delete bodyRange;
}

/*! @brief store tree level of each cell into separate array
 *
 * @param[in]  numCells
 * @param[in]  sourceCells
 * @param[out] sourceCells2  copy of @p sourceCells
 * @param[out] key           tree level of each cell
 * @param[out] value         enumeration of cell indices (std::iota)
 */
__global__ void getKeys(const int numCells, const CellData* sourceCells, CellData* sourceCells2, int* key, int* value)
{
    const int cellIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellIdx >= numCells) return;
    const CellData cell   = sourceCells[cellIdx];
    key[cellIdx]          = cell.level();
    value[cellIdx]        = cellIdx;
    sourceCells2[cellIdx] = cell;
}

/*! @brief compute first and last cell index for each tree level
 *
 * @param[in]  numCells
 * @param[in]  levels
 * @param[out] levelRange
 */
__global__ void getLevelRange(const int numCells, const int* levels, int2* levelRange)
{
    const int cellIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellIdx >= numCells) return;
    const int nextCellIdx = min(cellIdx + 1, numCells - 1);
    const int prevCellIdx = max(cellIdx - 1, 0);
    const int level       = levels[cellIdx];
    if (levels[prevCellIdx] < level || cellIdx == 0) levelRange[level].x = cellIdx;
    if (level < levels[nextCellIdx] || cellIdx == numCells - 1) levelRange[level].y = cellIdx + 1;
}

/*! @brief invert the given permutation
 *
 * @param[in]  numCells
 * @param[in]  value     input permutation
 * @param[out] key       output for the inverse permutation
 */
__global__ void getPermutation(const int numCells, const int* value, int* key)
{
    const int newIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (newIdx >= numCells) return;
    const int oldIdx = value[newIdx];
    key[oldIdx]      = newIdx;
}

/*! @brief gather/scatter sourceCells2 into sourceCells
 *
 * @param[in]  numCells
 * @param[in]  value          forward reording (scatter)
 * @param[in]  key            inverse ordering (gather)
 * @param[in]  sourceCells2
 * @param[out] sourceCells
 */
__global__ void permuteCells(const int numCells, const int* value, const int* key, const CellData* sourceCells2,
                             CellData* sourceCells)
{
    const int cellIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellIdx >= numCells) return;
    const int mapIdx = value[cellIdx];
    CellData cell    = sourceCells2[mapIdx];
    if (cell.isNode()) cell.setChild(key[cell.child()]);
    if (cell.parent() > 0) cell.setParent(key[cell.parent()]);
    sourceCells[cellIdx] = cell;
    if (cellIdx == 0) numLeafsGlob = 0;
}
} // namespace

class Build
{
public:
    /*! @brief launch tree build kernels
     *
     * @tparam     NCRIT         max number of particles per leaf
     * @param[in]  bodyPos       input particle positions
     * @param[out] bodyPos2
     * @param[out] box          coordinate bounding box, will be calculated from bodies
     * @param[out] levelRange   contains a pair of first and last cell for each tree level
     * @param[out] sourceCells  Array of tree cells, sorted according to subdivision level, then SFC key.
     *                          For each cell: contains level, parent, child, numChildren, first body, numBodies.
     * @return
     */
    template<int NCRIT>
    static int3 tree(cudaVec<fvec4>& bodyPos, cudaVec<fvec4>& bodyPos2, Box& box, cudaVec<int2>& levelRange,
                     cudaVec<CellData>& sourceCells)
    {
        const int numBodies = bodyPos.size();

        //! maximum estimated number of nodes
        const int maxNode   = numBodies / 10;

        //! space (plus temp space) for global bounds computation
        cudaVec<Bounds> bounds(2 * NTHREAD);

        cudaVec<int> octantSizePool(8 * maxNode);
        cudaVec<int> octantSizeScanPool(8 * maxNode);
        cudaVec<int> subOctantSizeScanPool(64 * maxNode);
        cudaVec<int> blockCounterPool(maxNode);
        cudaVec<int2> bodyRangePool(maxNode);
        cudaVec<int> key(numBodies);
        cudaVec<int> value(numBodies);

        //! Stack size ~ maxNode
        fprintf(stdout, "Stack size           : %g MB\n", 83 * maxNode * sizeof(int) / 1024.0 / 1024.0);
        //! Cell data ~ numBodies
        fprintf(stdout, "Cell data            : %g MB\n", numBodies * sizeof(CellData) / 1024.0 / 1024.0);
        cudaDeviceSynchronize();

        auto t0 = std::chrono::high_resolution_clock::now();
        //! compute coordinate bounding box
        getBounds<<<NTHREAD, NTHREAD>>>(numBodies, bounds.d(), bodyPos.d());
        kernelSuccess("getBounds");
        auto t1 = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        fprintf(stdout, "Get bounds           : %.7f s\n", dt);

        CUDA_SAFE_CALL(cudaMemcpyToSymbol(maxNodesGlob, &maxNode, sizeof(int), 0, cudaMemcpyHostToDevice));
        octantSizePool.zeros();
        octantSizeScanPool.zeros();
        subOctantSizeScanPool.zeros();
        blockCounterPool.zeros();
        bodyRangePool.zeros();
        cudaDeviceSynchronize();

        t0 = std::chrono::high_resolution_clock::now();

        //! count bodies in first 8 octants (children of root)
        getRootOctantSize<<<NTHREAD, NTHREAD>>>(numBodies, octantSizePool.d(), bodyPos.d());

        //! optimized kernel config
        CUDA_SAFE_CALL(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 16384));
        CUDA_SAFE_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
        CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&buildOctant<NCRIT, true>, cudaFuncCachePreferShared));
        CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&buildOctant<NCRIT, false>, cudaFuncCachePreferShared));

        buildOctree<NCRIT><<<1, 1>>>(numBodies,
                                     sourceCells.d(),
                                     octantSizePool.d(),
                                     octantSizeScanPool.d(),
                                     subOctantSizeScanPool.d(),
                                     blockCounterPool.d(),
                                     bodyRangePool.d(),
                                     reinterpret_cast<float4*>(bodyPos.d()),
                                     reinterpret_cast<float4*>(bodyPos2.d()));
        kernelSuccess("buildOctree");

        t1 = std::chrono::high_resolution_clock::now();
        dt = std::chrono::duration<double>(t1 - t0).count();
        fprintf(stdout, "Grow tree            : %.7f s\n", dt);

        int numLevels, numCells, numLeafs;
        CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&box, boxGlob, sizeof(float4)));
        CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&numLevels, numLevelsGlob, sizeof(int)));
        CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&numCells, numCellsGlob, sizeof(int)));
        CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&numLeafs, numLeafsGlob, sizeof(int)));
        cudaDeviceSynchronize();

        t0 = std::chrono::high_resolution_clock::now();
        int NBLOCK = (numCells - 1) / NTHREAD + 1;

        cudaVec<CellData> sourceCells2(numCells);

        getKeys<<<NBLOCK, NTHREAD>>>(numCells, sourceCells.d(), sourceCells2.d(), key.d(), value.d());
        kernelSuccess("getKeys");

        //! key:   contains level of each cell, will be sorted
        //! value: will contain the ordering that sorts key
        sort(numCells, key.d(), value.d());

        getLevelRange<<<NBLOCK, NTHREAD>>>(numCells, key.d(), levelRange.d());
        kernelSuccess("getLevelRange");

        getPermutation<<<NBLOCK, NTHREAD>>>(numCells, value.d(), key.d());
        kernelSuccess("getPermutation");

        sourceCells.alloc(numCells);

        permuteCells<<<NBLOCK, NTHREAD>>>(numCells, value.d(), key.d(), sourceCells2.d(), sourceCells.d());
        kernelSuccess("permuteCells");

        t1 = std::chrono::high_resolution_clock::now();
        dt = std::chrono::duration<double>(t1 - t0).count();
        fprintf(stdout, "Link tree            : %.7f s\n", dt);

        int3 counts = {numLevels, numCells, numLeafs};

        return counts;
    }
};

