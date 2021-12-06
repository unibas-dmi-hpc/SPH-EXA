#pragma once

#include "types.h"

namespace ryoanji
{

__device__ __forceinline__ int countLeadingZeros(uint32_t x)
{
    return __clz(x);
}

__device__ __forceinline__ int countLeadingZeros(uint64_t x)
{
    return __clzll(x);
}

__device__ __forceinline__ uint32_t reverseBits(uint32_t x)
{
    return __brev(x);
}

__device__ __forceinline__ uint64_t reverseBits(uint64_t x)
{
    return __brevll(x);
}

__device__ __forceinline__ int popCount(int x)
{
    return __popc(x);
}

__device__ __forceinline__ int popCount(long long int x)
{
    return __popcll(x);
}

//! @brief Compatibility wrapper for AMD. Note: do not HIPify!
template<class T>
__device__ __forceinline__ T shflSync(T value, int srcLane)
{
#ifdef __CUDACC__
    return __shfl_sync(0xFFFFFFFF, value, srcLane);
#else
    return __shfl(value, distance);
#endif
}

//! @brief Compatibility wrapper for AMD. Note: do not HIPify!
template<class T>
__device__ __forceinline__ T shflXorSync(T value, int width)
{
#ifdef __CUDACC__
    return __shfl_xor_sync(0xFFFFFFFF, value, width);
#else
    return __shfl_xor(value, width);
#endif
}

//! @brief Compatibility wrapper for AMD. Note: do not HIPify!
template<class T>
__device__ __forceinline__ T shflUpSync(T value, int distance)
{
#ifdef __CUDACC__
    return __shfl_up_sync(0xFFFFFFFF, value, distance);
#else
    return __shfl_up(value, distance);
#endif
}

//! @brief Compatibility wrapper for AMD. Note: do not HIPify!
__device__ __forceinline__ GpuConfig::ThreadMask ballotSync(bool flag)
{
#ifdef __CUDACC__
    return __ballot_sync(0xFFFFFFFF, flag);
#else
    return __ballot(flag);
#endif
}

//! @brief compute warp-wide min
template<class T>
__device__ __forceinline__ T warpMin(T laneVal)
{
    #pragma unroll
    for (int i = 0; i < GpuConfig::warpSizeLog2; i++)
    {
        laneVal = min(laneVal, shflXorSync(laneVal, 1 << i));
    }

    return laneVal;
}

//! @brief compute warp-wide max
template<class T>
__device__ __forceinline__ T warpMax(T laneVal)
{
    #pragma unroll
    for (int i = 0; i < GpuConfig::warpSizeLog2; i++)
    {
        laneVal = max(laneVal, shflXorSync(laneVal, 1 << i));
    }

    return laneVal;
}

//! @brief standard inclusive warp-scan
__device__ __forceinline__ int inclusiveScanInt(int value)
{
    unsigned lane = threadIdx.x & (GpuConfig::warpSize - 1);
    #pragma unroll
    for (int i = 1; i < GpuConfig::warpSize; i *= 2)
    {
        int partial = shflUpSync(value, i);
        if (i <= lane)
        {
            value += partial;
        }
    }
    return value;
}

//! @brief returns a mask with bits set for each warp lane before the calling lane
__device__ __forceinline__ GpuConfig::ThreadMask lanemask_lt()
{
    GpuConfig::ThreadMask lane = threadIdx.x & (GpuConfig::warpSize - 1);
    return (1 << lane) - 1;
    //int mask;
    //asm("mov.u32 %0, %lanemask_lt;" : "=r"(mask));
    //return mask;
}

__device__ __forceinline__ int exclusiveScanBool(const bool p)
{
    using SignedMask = std::make_signed_t<GpuConfig::ThreadMask>;
    GpuConfig::ThreadMask b = ballotSync(p);
    return popCount(SignedMask(b & lanemask_lt()));
}

__device__ __forceinline__ int reduceBool(const bool p)
{
    using SignedMask = std::make_signed_t<GpuConfig::ThreadMask>;
    GpuConfig::ThreadMask b = ballotSync(p);
    return popCount(SignedMask(b));
}

// Segmented scan int

//! @brief returns a mask with bits set for each warp lane before and including the calling lane
__device__ __forceinline__ GpuConfig::ThreadMask lanemask_le()
{
    GpuConfig::ThreadMask lane = threadIdx.x & (GpuConfig::warpSize - 1);
    return (2 << lane) - 1;
    //int mask;
    //asm("mov.u32 %0, %lanemask_le;" : "=r"(mask));
    //return mask;
}

/*! @brief perform range-limited inclusive warp scan
 *
 * @param value     scan input per lane
 * @param distance  number of preceding lanes to include in scanned result
 *                  distance has to be <= the current laneIdx
 * @return          the scanned value
 *
 * Due to the @p distance argument, the values of preceding lanes are only added
 * to the current lane if the source lane is less than @p distance away from the current lane.
 * If distance == current laneIdx, then the result is the same as a normal inclusive scan.
 */
__device__ __forceinline__ int inclusiveSegscan(int value, int distance)
{
    // distance should be less-equal the lane index
    assert (distance <= (threadIdx.x & (GpuConfig::warpSize - 1)));
    #pragma unroll
    for (int i = 1; i < GpuConfig::warpSize; i *= 2)
    {
        int partial = shflUpSync(value, i);
        if (i <= distance)
        {
            value += partial;
        }
    }

    return value;
}

/*! @brief performs an inclusive segmented warp prefix scan
 *
 * @param[in] packedValue input value per lane
 * @param[in] carryValue  offset for first prefix sum segment
 * @return                the prefix sum value per lane
 *
 * If packedValue is positive, behavior is like a usual prefix sum
 * Each negative packedValue will start a new prefix sum segment with a value of -packedValue - 1
 */
__device__ __forceinline__ int inclusiveSegscanInt(const int packedValue, const int carryValue)
{
    int laneIdx  = int(threadIdx.x) & (GpuConfig::warpSize - 1);

    int isNegative = packedValue < 0;
    int mask = -isNegative;

    // value = packedValue if packedValue >= 0
    // value = -packedValue - 1 if packedValue < 0
    int value = (~mask & packedValue) + (mask & (-1 - packedValue));

    GpuConfig::ThreadMask flags = ballotSync(isNegative);

    // distance = number of preceding lanes to include in scanned value
    // e.g. if distance = 0, then no preceding lane value will be added to scannedValue
    int distance     = countLeadingZeros(flags & lanemask_le()) + laneIdx - (GpuConfig::warpSize - 1);
    int scannedValue = inclusiveSegscan(value, min(distance, laneIdx));

    // the lowest lane index for which packedValue was negative, warpSize if all were positive
    // __brev reverses the bit order
    int firstNegativeLane = countLeadingZeros(reverseBits(flags));
    int addCarry          = -(laneIdx < firstNegativeLane);

    return scannedValue + (carryValue & addCarry);
}

/*! @brief warp stream compaction, warp-vote part
 *
 * @param[in]  flag            compaction flag, keep laneVal yes/no
 * @param[in]  fillLevel       current fill level of @p queue
 * @param[out] numElementsKeep output for number of valid flags in the warp
 * @return                     the new queue storage position for laneVal, can exceed warp size
 *                             due to non-zero fillLevel prior to call
 *
 * Note: incrementing fillLevel by numElementsKeep has to happen after calling warpExchange
 * to allow warp exchange for multiple queues.
 */
__device__ __forceinline__ int warpCompact(bool flag, const int fillLevel, int* numElementsKeep)
{
    using SignedMask = std::make_signed_t<GpuConfig::ThreadMask>;

    GpuConfig::ThreadMask keepBallot = ballotSync(flag);
    int laneCompacted   = fillLevel + popCount(SignedMask(keepBallot & lanemask_lt())); // exclusive scan of keepBallot
    *numElementsKeep    = popCount(SignedMask(keepBallot));

    return laneCompacted;
}

/*! @brief warp stream compaction
 *
 * @tparam       T             a builtin type like int or float
 * @param[inout] queue         the warp queue, distinct T element for each lane
 * @param[in]    laneVal       input value per lane
 * @param[in]    flag          compaction flag, keep laneVal yes/no
 * @param[in]    laneCompacted new lane position for @p laneVal if @p flag is true
 * @param[in]    fillLevel     current fill level of @p queue, lanes below already have valid elements in @p queue
 * @param[-]     sm_exchange   shared memory buffer for intra warp-exchange, length=WarpSize, uninitialized
 *
 * This moves laneVal elements inside the warp into queue via the shared sm_exchange buffer
 */
template<class T>
__device__ __forceinline__ void warpExchange(T* queue, const T* laneVal, bool flag, int laneCompacted,
                                             const int fillLevel, volatile T* sm_exchange)
{
    int laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);

    // if laneVal is valid and queue has space
    if (flag && laneCompacted >= 0 && laneCompacted < GpuConfig::warpSize) { sm_exchange[laneCompacted] = *laneVal; }
    __syncwarp();

    // pull newly compacted elements into the queue
    // note sm_exchange is uninitialized, therefore we cannot pull down lanes that we did not set above
    // i.e. below the current fill level
    if (laneIdx >= fillLevel) { *queue = sm_exchange[laneIdx]; }
    __syncwarp();
}

} // namespace ryoanji
