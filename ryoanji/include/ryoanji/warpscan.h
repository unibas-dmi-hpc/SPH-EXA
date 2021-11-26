#pragma once

namespace
{

__device__ __forceinline__ void getMinMax(fvec3& _Xmin, fvec3& _Xmax, const fvec3& pos)
{
    fvec3 Xmin = pos;
    fvec3 Xmax = Xmin;

#pragma unroll
    for (int i = 0; i < WARP_SIZE2; i++)
    {
        Xmin[0] = min(Xmin[0], __shfl_xor_sync(0xFFFFFFFF, Xmin[0], 1 << i));
        Xmin[1] = min(Xmin[1], __shfl_xor_sync(0xFFFFFFFF, Xmin[1], 1 << i));
        Xmin[2] = min(Xmin[2], __shfl_xor_sync(0xFFFFFFFF, Xmin[2], 1 << i));
        Xmax[0] = max(Xmax[0], __shfl_xor_sync(0xFFFFFFFF, Xmax[0], 1 << i));
        Xmax[1] = max(Xmax[1], __shfl_xor_sync(0xFFFFFFFF, Xmax[1], 1 << i));
        Xmax[2] = max(Xmax[2], __shfl_xor_sync(0xFFFFFFFF, Xmax[2], 1 << i));
    }

    _Xmin[0] = min(_Xmin[0], Xmin[0]);
    _Xmin[1] = min(_Xmin[1], Xmin[1]);
    _Xmin[2] = min(_Xmin[2], Xmin[2]);
    _Xmax[0] = max(_Xmax[0], Xmax[0]);
    _Xmax[1] = max(_Xmax[1], Xmax[1]);
    _Xmax[2] = max(_Xmax[2], Xmax[2]);
}

// Scan int

//! @brief standard inclusive warp-scan
__device__ __forceinline__ int inclusiveScanInt(int value)
{
    unsigned lane = threadIdx.x & (WARP_SIZE - 1);
    #pragma unroll
    for (int i = 1; i < WARP_SIZE; i *= 2)
    {
        int partial = __shfl_up_sync(0xFFFFFFFF, value, i);
        if (i <= lane)
        {
            value += partial;
        }
    }
    return value;
}

// Scan bool

//! @brief returns a mask with bits set for each warp lane before the calling lane
__device__ __forceinline__ int lanemask_lt()
{
    unsigned lane = threadIdx.x & (WARP_SIZE - 1);
    return (1 << lane) - 1;
    //int mask;
    //asm("mov.u32 %0, %lanemask_lt;" : "=r"(mask));
    //return mask;
}

__device__ __forceinline__ int exclusiveScanBool(const bool p)
{
    unsigned b = __ballot_sync(0xFFFFFFFF, p);
    return __popc(b & lanemask_lt());
}

__device__ __forceinline__ int reduceBool(const bool p)
{
    unsigned b = __ballot_sync(0xFFFFFFFF, p);
    return __popc(b);
}

// Segmented scan int

//! @brief returns a mask with bits set for each warp lane before and including the calling lane
__device__ __forceinline__ int lanemask_le()
{
    unsigned lane = threadIdx.x & (WARP_SIZE - 1);
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
    assert (distance <= (threadIdx.x & (WARP_SIZE- 1)));
    #pragma unroll
    for (int i = 1; i < WARP_SIZE; i *= 2)
    {
        int partial = __shfl_up_sync(0xFFFFFFFF, value, i);
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
    int laneIdx  = int(threadIdx.x) & (WARP_SIZE - 1);

    int isNegative = packedValue < 0;
    int mask = -isNegative;

    // value = packedValue if packedValue >= 0
    // value = -packedValue - 1 if packedValue < 0
    int value = (~mask & packedValue) + (mask & (-1 - packedValue));

    int flags = int(__ballot_sync(0xFFFFFFFF, isNegative));

    // distance = number of preceding lanes to include in scanned value
    // e.g. if distance = 0, then no preceding lane value will be added to scannedValue
    int distance     = __clz(flags & lanemask_le()) + laneIdx - (WARP_SIZE - 1);
    int scannedValue = inclusiveSegscan(value, min(distance, laneIdx));

    // the lowest lane index for which packedValue was negative, WARP_SIZE if all were positive
    // __brev reverses the bit order
    int firstNegativeLane = __clz(int(__brev(flags)));
    int addCarry          = -(laneIdx < firstNegativeLane);

    return scannedValue + (carryValue & addCarry);
}

} // namespace

