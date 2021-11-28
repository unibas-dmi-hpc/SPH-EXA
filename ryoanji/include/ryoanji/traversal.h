#pragma once
#include <algorithm>
#include "kernel.h"
#include "warpscan.h"

#define IF(x) (-(int)(x))

struct GpuConfig
{
    //! @brief number of threads per warp
    static constexpr int warpSize = 32;
    //! @brief log2(warpSize)
    static constexpr int warpSizeLog2 = 5;
    //! @brief number of multiprocessors
    static int smCount;
};

int GpuConfig::smCount = 56;

struct TravConfig
{
    //! @brief size of global workspace memory per warp
    static constexpr int memPerWarp = 2048 * GpuConfig::warpSize;
    //! @brief number of threads per block for the traversal kernel
    static constexpr int numThreadsTraversal = 256;
    //! @brief log2(numThreadsTraversal)
    static constexpr int numThreadsTraversalLog2 = 8;

    static constexpr int numWarpsPerSm = 16;
    //! @brief maximum number of simultaneously active blocks
    static int maxNumActiveBlocks;

    //! @brief number of particles per target, i.e. per warp
    static constexpr int targetSize = 64;
};

int TravConfig::maxNumActiveBlocks =
    GpuConfig::smCount * TravConfig::numWarpsPerSm / (TravConfig::numThreadsTraversal / GpuConfig::warpSize);

namespace
{
__device__ __forceinline__ int ringAddr(const int i) { return i & (TravConfig::memPerWarp - 1); }

__host__ __device__ __forceinline__ bool applyMAC(fvec3 sourceCenter, float MAC, CellData sourceData,
                                                  fvec3 targetCenter, fvec3 targetSize)
{
    fvec3 dX = abs(targetCenter - sourceCenter) - targetSize;
    dX += abs(dX);
    dX *= 0.5f;
    const float R2 = norm(dX);
    return R2 < fabsf(MAC) || sourceData.nbody() < 3;
}

__device__ void approxAcc(fvec4 acc_i[2], const fvec3 pos_i[2], const int cellIdx, const fvec4* __restrict__ srcCenter,
                          const fvec4* __restrict__ Multipoles, float EPS2)
{
    fvec4 M4[NVEC4];
    float M[4 * NVEC4];
    fvec4 Xj(0.0f);
    if (cellIdx >= 0)
    {
        Xj = srcCenter[cellIdx];
        #pragma unroll
        for (int i = 0; i < NVEC4; i++)
            M4[i] = Multipoles[NVEC4 * cellIdx + i];
    }
    else
    {
        #pragma unroll
        for (int i = 0; i < NVEC4; i++)
            M4[i] = 0.0f;
    }
    for (int j = 0; j < GpuConfig::warpSize; j++)
    {
        const fvec3 pos_j(__shfl_sync(0xFFFFFFFF, Xj[0], j),
                          __shfl_sync(0xFFFFFFFF, Xj[1], j),
                          __shfl_sync(0xFFFFFFFF, Xj[2], j));
        #pragma unroll
        for (int i = 0; i < NVEC4; i++)
        {
            M[4 * i + 0] = __shfl_sync(0xFFFFFFFF, M4[i][0], j);
            M[4 * i + 1] = __shfl_sync(0xFFFFFFFF, M4[i][1], j);
            M[4 * i + 2] = __shfl_sync(0xFFFFFFFF, M4[i][2], j);
            M[4 * i + 3] = __shfl_sync(0xFFFFFFFF, M4[i][3], j);
        }
        for (int k = 0; k < 2; k++)
            acc_i[k] = M2P(acc_i[k], pos_i[k], pos_j, *(fvecP*)M, EPS2);
    }
}

/*! @brief traverse one warp with up to 64 target bodies down the tree
 *
 * @param[inout] acc_i         acceleration to add to, two fvec4 per lane
 * @param[-]     M4            shared mem for 1 multipole in fvec4 format, uninitialized
 * @param[-]     M             shared mem for 1 multipole in float format, uninitialized
 * @param[in]    pos_i         target positions, 2 per lane
 * @param[in]    targetCenter  geometrical target center
 * @param[in]    targetSize    geometrical target bounding box size
 * @param[in]    bodyPos       source bodies as referenced by tree cells
 * @param[in]    srcCells      source cell data
 * @param[in]    srcCenter     source center data, x,y,z location and square of MAC radius, same order as srcCells
 * @param[in]    Multipoles    the multipole expansions in the same order as srcCells
 * @param[in]    EPS2          plummer softening
 * @param[in]    rootRange     source cell indices indices of the top 8 octants
 * @param[-]     tempQueue     shared mem int pointer to 32 ints, uninitialized
 * @param[-]     cellQueue     shared mem int pointer to global memory, 4096 ints per thread, uninitialized
 * @return
 */
__device__ uint2 traverseWarp(fvec4* acc_i, const fvec3 pos_i[2], const fvec3 targetCenter, const fvec3 targetSize,
                              const fvec4* __restrict__ bodyPos, const CellData* __restrict__ srcCells, const fvec4* __restrict__ srcCenter,
                              const fvec4* __restrict__ Multipoles, float EPS2, int2 rootRange, volatile int* tempQueue,
                              int* cellQueue)
{
    const int laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);

    uint2 counters = {0, 0};
    int approxQueue, directQueue;

    // populate initial cell queue
    for (int root = rootRange.x; root < rootRange.y; root += GpuConfig::warpSize)
    {
        if (root + laneIdx < rootRange.y)
        {
            cellQueue[ringAddr(root - rootRange.x + laneIdx)] = root + laneIdx;
        }
    }

    // these variables are always identical on all warp lanes
    int numSources   = rootRange.y - rootRange.x; // current stack size
    int newSources   = 0; // stack size for next level
    int oldSources   = 0; // cell indices done
    int sourceOffset = 0; // current level stack pointer, once this reaches numSources, the level is done
    int approxOffset = 0;
    int bodyOffset   = 0;

    while (numSources > 0) // While there are source cells to traverse
    {
        const int sourceIdx   = sourceOffset + laneIdx;                      // Source cell index of current lane
        const int sourceQueue = cellQueue[ringAddr(oldSources + sourceIdx)]; // Global source cell index in queue
        const fvec4 MAC       = srcCenter[sourceQueue];                      // load source cell center + MAC
        const fvec3 sourceCenter(MAC[0], MAC[1], MAC[2]);                    // Source cell center
        const CellData sourceData = srcCells[sourceQueue];                   // load source cell data
        const bool isNode         = sourceData.isNode();                     // Is non-leaf cell
        const bool isClose =
            applyMAC(sourceCenter, MAC[3], sourceData, targetCenter, targetSize); // Is too close for MAC
        const bool isSource = sourceIdx < numSources;                             // Source index is within bounds

        // Split
        const bool isSplit     = isNode && isClose && isSource;        // Source cell must be split
        const int childBegin   = sourceData.child();                   // First child cell
        const int numChild     = sourceData.nchild() & IF(isSplit);    // Number of child cells (masked by split flag)
        const int numChildScan = inclusiveScanInt(numChild);           // Inclusive scan of numChild
        const int numChildLane = numChildScan - numChild;              // Exclusive scan of numChild
        const int numChildWarp = __shfl_sync(0xFFFFFFFF, numChildScan, // Total numChild of current warp
                                             GpuConfig::warpSize - 1);
        sourceOffset += min(GpuConfig::warpSize, numSources - sourceOffset);     // advance current level stack pointer
        if (numChildWarp + numSources - sourceOffset > TravConfig::memPerWarp)   // If cell queue overflows
            return make_uint2(0xFFFFFFFF, 0xFFFFFFFF);                 // Exit kernel
        int childIdx = oldSources + numSources + newSources + numChildLane; // Child index of current lane
        for (int i = 0; i < numChild; i++)                                  // Loop over child cells for each lane
            cellQueue[ringAddr(childIdx + i)] = childBegin + i;             // Queue child cells for next level
        newSources += numChildWarp; //  Increment source cell count for next loop

        // Approx
        const bool isApprox     = !isClose && isSource;                 // Source cell can be used for M2P
        const uint approxBallot = __ballot_sync(0xFFFFFFFF, isApprox);  // Gather approx flags
        const int numApproxLane = __popc(approxBallot & lanemask_lt()); // Exclusive scan of approx flags
        const int numApproxWarp = __popc(approxBallot);                 // Total isApprox for current warp
        int approxIdx           = approxOffset + numApproxLane;         // Approx cell index of current lane
        tempQueue[laneIdx]      = approxQueue;                          // Fill queue with remaining sources for approx
        __syncwarp();
        if (isApprox && approxIdx < GpuConfig::warpSize)         // If approx flag is true and index is within bounds
        {
            tempQueue[approxIdx] = sourceQueue;        // Fill approx queue with current sources
        }
        __syncwarp();
        if (approxOffset + numApproxWarp >= GpuConfig::warpSize) // If approx queue is larger than the warp size
        {
            // Call M2P kernel
            approxAcc(acc_i, pos_i, tempQueue[laneIdx], srcCenter, Multipoles, EPS2);

            approxOffset -= GpuConfig::warpSize;      // Decrement approx queue size
            approxIdx = approxOffset + numApproxLane; // Update approx index using new queue size
            if (isApprox && approxIdx >= 0)           // If approx flag is true and index is within bounds
                tempQueue[approxIdx] = sourceQueue;   // Fill approx queue with current sources
            counters.x += GpuConfig::warpSize;        // Increment M2P counter
        }                                             // End if for approx queue size
        __syncwarp();
        approxQueue = tempQueue[laneIdx];             // Free temp queue for use in direct
        approxOffset += numApproxWarp;                // Increment approx queue offset

        // Direct
        const bool isLeaf       = !isNode;                               // Is leaf cell
        bool isDirect           = isClose && isLeaf && isSource;         // Source cell can be used for P2P
        const int bodyBegin     = sourceData.body();                     // First body in cell
        const int numBodies     = sourceData.nbody() & IF(isDirect);     // Number of bodies in cell
        const int numBodiesScan = inclusiveScanInt(numBodies);           // Inclusive scan of numBodies
        int numBodiesLane       = numBodiesScan - numBodies;             // Exclusive scan of numBodies
        int numBodiesWarp       = __shfl_sync(0xFFFFFFFF, numBodiesScan, // Total numBodies of current warp
                                              GpuConfig::warpSize - 1);
        int tempOffset = 0;         //  Initialize temp queue offset
        while (numBodiesWarp > 0)   // While there are bodies to process
        {
            tempQueue[laneIdx] = 1; // Initialize body queue
            if (isDirect && (numBodiesLane < GpuConfig::warpSize))
            {                                              // If direct flag is true and index is within bounds
                isDirect                 = false;          // Set flag as processed
                tempQueue[numBodiesLane] = -1 - bodyBegin; // Put body in queue
            }                                              // End if for direct flag
            const int bodyQueue =
                inclusiveSegscanInt(tempQueue[laneIdx], tempOffset);        // Inclusive segmented scan of temp queue
            tempOffset = __shfl_sync(0xFFFFFFFF, bodyQueue, GpuConfig::warpSize - 1); // Last lane has the temp queue offset

            if (numBodiesWarp >= GpuConfig::warpSize) // If warp is full of bodies
            {
                const fvec4 pos = bodyPos[bodyQueue]; // Load position of source bodies
                for (int j = 0; j < GpuConfig::warpSize; j++)
                {                                                             // Loop over the warp size
                    const fvec3 pos_j(__shfl_sync(0xFFFFFFFF, pos[0], j),     // Get source x value from lane j
                                      __shfl_sync(0xFFFFFFFF, pos[1], j),     // Get source y value from lane j
                                      __shfl_sync(0xFFFFFFFF, pos[2], j));    // Get source z value from lane j
                    const float q_j = __shfl_sync(0xFFFFFFFF, pos[3], j);     // Get source w value from lane j
                    #pragma unroll                                            // Unroll loop
                    for (int k = 0; k < 2; k++)                               // Loop over two targets
                        acc_i[k] = P2P(acc_i[k], pos_i[k], pos_j, q_j, EPS2); // Call P2P kernel
                }                                                             // End loop over the warp size
                numBodiesWarp -= GpuConfig::warpSize;                                   // Decrement body queue size
                numBodiesLane -= GpuConfig::warpSize;                                   // Derecment lane offset of body index
                counters.y += GpuConfig::warpSize;                                      // Increment P2P counter
            }
            else // If warp is not entirely full of bodies
            {
                int bodyIdx        = bodyOffset + laneIdx; // Body index of current lane
                tempQueue[laneIdx] = directQueue;          // Initialize body queue with saved values
                if (bodyIdx < GpuConfig::warpSize)         // If body index is less than the warp size
                    tempQueue[bodyIdx] = bodyQueue;        // Push bodies into queue
                bodyOffset += numBodiesWarp;               // Increment body queue offset
                if (bodyOffset >= GpuConfig::warpSize)               // If this causes the body queue to spill
                {
                    const fvec4 pos = bodyPos[tempQueue[laneIdx]]; // Load position of source bodies
                    for (int j = 0; j < GpuConfig::warpSize; j++)
                    {                                                          // Loop over the warp size
                        const fvec3 pos_j(__shfl_sync(0xFFFFFFFF, pos[0], j),  // Get source x value from lane j
                                          __shfl_sync(0xFFFFFFFF, pos[1], j),  // Get source y value from lane j
                                          __shfl_sync(0xFFFFFFFF, pos[2], j)); // Get source z value from lane j
                        const float q_j = __shfl_sync(0xFFFFFFFF, pos[3], j);  // Get source w value from lane j
                        #pragma unroll
                        for (int k = 0; k < 2; k++)                               // Loop over two targets
                            acc_i[k] = P2P(acc_i[k], pos_i[k], pos_j, q_j, EPS2); // Call P2P kernel
                    }                                                             // End loop over the warp size
                    bodyOffset -= GpuConfig::warpSize;                            // Decrement body queue size
                    bodyIdx -= GpuConfig::warpSize;     // Decrement body index of current lane
                    if (bodyIdx >= 0)                   // If body index is valid
                        tempQueue[bodyIdx] = bodyQueue; // Push bodies into queue
                    counters.y += GpuConfig::warpSize;  // Increment P2P counter
                }                                       // End if for body queue spill
                directQueue   = tempQueue[laneIdx];     // Free temp queue for use in approx
                numBodiesWarp = 0;                      // Reset numBodies of current warp
            }                                           // End if for warp full of bodies
        }                                               // End while loop for bodies to process

        //  If the current level is done
        if (sourceOffset >= numSources)
        {
            oldSources += numSources;      // Update finished source size
            numSources   = newSources;     // Update current source size
            sourceOffset = newSources = 0; // Initialize next source size and offset
        }                                  // End if for level finalization
    }                                      // End while for source cells to traverse

    if (approxOffset > 0) // If there are leftover approx cells
    {
        // Call M2P kernel
        approxAcc(acc_i, pos_i, laneIdx < approxOffset ? approxQueue : -1, srcCenter, Multipoles, EPS2);

        counters.x += approxOffset; // Increment M2P counter
        approxOffset = 0;           // Reset offset for approx
    }                               // End if for leftover approx cells

    if (bodyOffset > 0) // If there are leftover direct cells
    {
        const int bodyQueue = laneIdx < bodyOffset ? directQueue : -1;  // Get body index
        const fvec4 pos     = bodyQueue >= 0 ? bodyPos[bodyQueue] :     // Load position of source bodies
                              make_float4(0.0f, 0.0f, 0.0f, 0.0f);      // With padding for invalid lanes
        for (int j = 0; j < GpuConfig::warpSize; j++)
        {                                                             // Loop over the warp size
            const fvec3 pos_j(__shfl_sync(0xFFFFFFFF, pos[0], j),     // Get source x value from lane j
                              __shfl_sync(0xFFFFFFFF, pos[1], j),     // Get source y value from lane j
                              __shfl_sync(0xFFFFFFFF, pos[2], j));    // Get source z value from lane j
            const float q_j = __shfl_sync(0xFFFFFFFF, pos[3], j);     // Get source w value from lane j
            #pragma unroll                                            // Unroll loop
            for (int k = 0; k < 2; k++)                               // Loop over two targets
                acc_i[k] = P2P(acc_i[k], pos_i[k], pos_j, q_j, EPS2); // Call P2P kernel
        }                                                             // End loop over the warp size
        counters.y += bodyOffset;                                     // Increment P2P counter
        bodyOffset = 0;                                               // Reset offset for direct
    }                                                                 // End if for leftover direct cells

    return counters; // Return M2P & P2P counters
}

__device__ uint64_t sumP2PGlob     = 0;
__device__ unsigned int maxP2PGlob = 0;
__device__ uint64_t sumM2PGlob     = 0;
__device__ unsigned int maxM2PGlob = 0;

__device__ unsigned int targetCounterGlob = 0;

__global__ void resetTraversalCounters()
{
    sumP2PGlob = 0;
    maxP2PGlob = 0;
    sumM2PGlob = 0;
    maxM2PGlob = 0;

    targetCounterGlob = 0;
}

/*! @brief tree traversal
 *
 * @param[in]  firstBody     index of first body in bodyPos to compute acceleration for
 * @param[in]  lastBody      index (exclusive) of last body in @p bodyPos to compute acceleration for
 * @param[in]  images        number of periodic images to include
 * @param[in]  EPS2          Plummer softening
 * @param[in]  cycle         2 * M_PI
 * @param[in]  levelRange    (start,end) index pairs into texCell, texCenter and texSource for each tree level
 * @param[in]  bodyPos       pointer to SFC-sorted bodies
 * @param[in]  srcCells      source cell data
 * @param[in]  srcCenter     source center data, x,y,z location and square of MAC radius, same order as srcCells
 * @param[in]  Multipoles    the multipole expansions in the same order as srcCells
 * @param[out] bodyAcc       body accelerations
 * @param[-]   globalPool    length proportional to number of warps in the launch grid, uninitialized
 */
__global__ __launch_bounds__(TravConfig::numThreadsTraversal)
void traverse(int firstBody, int lastBody, int images, float EPS2, float cycle,
              const int2* levelRange, const fvec4* __restrict__ bodyPos,
              const CellData* __restrict__ srcCells,
              const fvec4* __restrict__ srcCenter, const fvec4* __restrict__ Multipoles,
              fvec4* bodyAcc, int* globalPool)
{
    const int laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);
    const int warpIdx = threadIdx.x >> GpuConfig::warpSizeLog2;

    const int numWarpsLog2  = TravConfig::numThreadsTraversalLog2 - GpuConfig::warpSizeLog2;
    const int numTargets    = (lastBody - firstBody - 1) / TravConfig::targetSize + 1;

    __shared__ int sharedPool[TravConfig::numThreadsTraversal];

    // warp-common shared mem, 1 int per thread
    int* tempQueue = sharedPool + GpuConfig::warpSize * warpIdx;
    // warp-common global mem storage, 4096 ints per thread
    int* cellQueue = globalPool + TravConfig::memPerWarp * ((blockIdx.x << numWarpsLog2) + warpIdx);

    //int targetIdx = (blockIdx.x << numWarpsLog2) + warpIdx;
    int targetIdx = 0;

    while (true)
    //for(; targetIdx < numTargets; targetIdx += (gridDim.x << numWarpsLog2))
    {
        // first thread in warp grabs next target
        if (laneIdx == 0)
        {
            targetIdx = atomicAdd(&targetCounterGlob, 1);
        }
        targetIdx = __shfl_sync(0xFFFFFFFF, targetIdx, 0, GpuConfig::warpSize);

        if (targetIdx >= numTargets) return;

        const int bodyBegin = firstBody + targetIdx * TravConfig::targetSize;
        const int bodyEnd   = min(bodyBegin + TravConfig::targetSize, lastBody);

        // load target coordinates
        fvec3 pos_i[2], pos_p[2];
        for (int i = 0; i < 2; i++)
        {
            int bodyIdx = min(bodyBegin + i * GpuConfig::warpSize + laneIdx, bodyEnd - 1);
            pos_i[i]    = make_fvec3(fvec4(bodyPos[bodyIdx]));
        }

        fvec3 Xmin = pos_i[0];
        fvec3 Xmax = Xmin;
        for (int i = 0; i < 2; i++)
        {
            getMinMax(Xmin, Xmax, pos_i[i]);
        }

        Xmin[0] = __shfl_sync(0xFFFFFFFF, Xmin[0], 0);
        Xmin[1] = __shfl_sync(0xFFFFFFFF, Xmin[1], 0);
        Xmin[2] = __shfl_sync(0xFFFFFFFF, Xmin[2], 0);
        Xmax[0] = __shfl_sync(0xFFFFFFFF, Xmax[0], 0);
        Xmax[1] = __shfl_sync(0xFFFFFFFF, Xmax[1], 0);
        Xmax[2] = __shfl_sync(0xFFFFFFFF, Xmax[2], 0);

        const fvec3 targetCenter = (Xmax + Xmin) * 0.5f;
        const fvec3 targetSize   = (Xmax - Xmin) * 0.5f;
        fvec4 acc_i[2]           = {0.0f, 0.0f};
        fvec3 Xperiodic          = 0.0f;

        int numP2P = 0, numM2P = 0;
        for (int ix = -images; ix <= images; ix++)
        {
            for (int iy = -images; iy <= images; iy++)
            {
                for (int iz = -images; iz <= images; iz++)
                {
                    Xperiodic[0]               = ix * cycle;
                    Xperiodic[1]               = iy * cycle;
                    Xperiodic[2]               = iz * cycle;
                    const fvec3 targetPeriodic = targetCenter - Xperiodic;
                    for (int i = 0; i < 2; i++)
                        pos_p[i] = pos_i[i] - Xperiodic;
                    const uint2 counters = traverseWarp(acc_i,
                                                        pos_p,
                                                        targetPeriodic,
                                                        targetSize,
                                                        bodyPos,
                                                        srcCells,
                                                        srcCenter,
                                                        Multipoles,
                                                        EPS2,
                                                        levelRange[1],
                                                        tempQueue,
                                                        cellQueue);
                    assert(!(counters.x == 0xFFFFFFFF && counters.y == 0xFFFFFFFF));
                    numM2P += counters.x;
                    numP2P += counters.y;
                }
            }
        }

        int maxP2P = numP2P;
        int sumP2P = 0;
        int maxM2P = numM2P;
        int sumM2P = 0;

        const int bodyIdx = bodyBegin + laneIdx;
        for (int i = 0; i < 2; i++)
        {
            if (i * GpuConfig::warpSize + bodyIdx < bodyEnd)
            {
                sumM2P += numM2P;
                sumP2P += numP2P;
            }
        }
        #pragma unroll
        for (int i = 0; i < GpuConfig::warpSizeLog2; i++)
        {
            maxP2P = max(maxP2P, __shfl_xor_sync(0xFFFFFFFF, maxP2P, 1 << i));
            sumP2P += __shfl_xor_sync(0xFFFFFFFF, sumP2P, 1 << i);
            maxM2P = max(maxM2P, __shfl_xor_sync(0xFFFFFFFF, maxM2P, 1 << i));
            sumM2P += __shfl_xor_sync(0xFFFFFFFF, sumM2P, 1 << i);
        }
        if (laneIdx == 0)
        {
            atomicMax(&maxP2PGlob, maxP2P);
            atomicAdd((unsigned long long*)&sumP2PGlob, (unsigned long long)sumP2P);
            atomicMax(&maxM2PGlob, maxM2P);
            atomicAdd((unsigned long long*)&sumM2PGlob, (unsigned long long)sumM2P);
        }

        for (int i = 0; i < 2; i++)
        {
            if (bodyIdx + i * GpuConfig::warpSize < bodyEnd)
            {
                bodyAcc[i * GpuConfig::warpSize + bodyIdx] = acc_i[i];
            }
        }
    }
}

} // namespace

class Traversal
{
public:
    /*! @brief
     *
     * @param[in]  firstBody     index of first body in @p bodyPos to compute acceleration for
     * @param[in]  lastBody      index (exclusive) of last body in @p bodyPos to compute acceleration for
     * @param[in]  images        number of periodic images (per direction per dimension)
     * @param[in]  eps           plummer softening parameter
     * @param[in]  cycle         2 * M_PI
     * @param[in]  bodyPos       bodies, in SFC order and as referenced by sourceCells
     * @param[out] bodyAcc       output body acceleration in SFC order
     * @param[in]  sourceCells   tree connectivity and body location cell data
     * @param[in]  sourceCenter  center-of-mass and MAC radius^2 for each cell
     * @param[in]  Multipole     cell multipoles
     * @param[in]  levelRange
     * @return
     */
    static fvec4 approx(int firstBody, int lastBody, int images, float eps, float cycle,
                        cudaVec<fvec4>& bodyPos, cudaVec<fvec4>& bodyAcc,
                        cudaVec<CellData>& sourceCells, cudaVec<fvec4>& sourceCenter,
                        cudaVec<fvec4>& Multipole, cudaVec<int2>& levelRange)
    {
        constexpr int numWarpsPerBlock = 1 << (TravConfig::numThreadsTraversalLog2 - GpuConfig::warpSizeLog2);

        int numBodies = lastBody - firstBody;

        // each target gets a warp (numWarps == numTargets)
        int numWarps  = (numBodies - 1) / TravConfig::targetSize + 1;
        int numBlocks = (numWarps - 1) / numWarpsPerBlock + 1;
        numBlocks = std::min(numBlocks, TravConfig::maxNumActiveBlocks);

        const int poolSize  = TravConfig::memPerWarp * numWarpsPerBlock * numBlocks;

        cudaVec<int> globalPool(poolSize);

        cudaDeviceSynchronize();

        resetTraversalCounters<<<1,1>>>();

        auto t0 = std::chrono::high_resolution_clock::now();
        CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&traverse, cudaFuncCachePreferL1));
        traverse<<<numBlocks, TravConfig::numThreadsTraversal>>>(firstBody,
                                                                lastBody,
                                                                images,
                                                                eps * eps,
                                                                cycle,
                                                                levelRange.d(),
                                                                bodyPos.d(),
                                                                sourceCells.d(),
                                                                sourceCenter.d(),
                                                                Multipole.d(),
                                                                bodyAcc.d(),
                                                                globalPool.d());
        kernelSuccess("traverse");

        auto t1  = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(t1 - t0).count();

        uint64_t sumP2P, sumM2P;
        unsigned int maxP2P, maxM2P;

        CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&sumP2P, sumP2PGlob, sizeof(uint64_t)));
        CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&maxP2P, maxP2PGlob, sizeof(unsigned int)));
        CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&sumM2P, sumM2PGlob, sizeof(uint64_t)));
        CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&maxM2P, maxM2PGlob, sizeof(unsigned int)));

        fvec4 interactions;
        interactions[0] = float(sumP2P) * 1.0f / float(numBodies);
        interactions[1] = float(maxP2P);
        interactions[2] = float(sumM2P) * 1.0f / float(numBodies);
        interactions[3] = float(maxM2P);

        float flops = (interactions[0] * 20.0f + interactions[2] * 2.0f * powf(P, 3)) * float(numBodies) / dt / 1e12f;

        fprintf(stdout, "Traverse             : %.7f s (%.7f TFlops)\n", dt, flops);

        return interactions;
    }
};

