#pragma once

#include <chrono>

#define NBITS 21

extern void sort(const int size, uint64_t* key, int* value);
extern void scan(const int size, uint64_t* key, int* value);

namespace
{

__device__ unsigned int numTargetGlob = 0;

__global__ void resetNumTargets()
{
    numTargetGlob = 0;
}

__host__ __device__ void swap(int& a, int& b)
{
    int c(a);
    a = b;
    b = c;
}

__host__ __device__ uint64_t getHilbert(int3 iX)
{
    const int octantMap[8] = {0, 1, 7, 6, 3, 2, 4, 5};
    int mask               = 1 << (NBITS - 1);
    uint64_t key           = 0;

    #pragma unroll
    for (int i = 0; i < NBITS; i++)
    {
        const int ix     = (iX.x & mask) ? 1 : 0;
        const int iy     = (iX.y & mask) ? 1 : 0;
        const int iz     = (iX.z & mask) ? 1 : 0;
        const int octant = (ix << 2) + (iy << 1) + iz;
        if (octant == 0) { swap(iX.y, iX.z); }
        else if (octant == 1 || octant == 5)
        {
            swap(iX.x, iX.y);
        }
        else if (octant == 4 || octant == 6)
        {
            iX.x = (iX.x) ^ (-1);
            iX.z = (iX.z) ^ (-1);
        }
        else if (octant == 3 || octant == 7)
        {
            iX.x = (iX.x) ^ (-1);
            iX.y = (iX.y) ^ (-1);
            swap(iX.x, iX.y);
        }
        else
        {
            iX.y = (iX.y) ^ (-1);
            iX.z = (iX.z) ^ (-1);
            swap(iX.y, iX.z);
        }
        key = (key << 3) + octantMap[octant];
        mask >>= 1;
    }
    return key;
}

/*! @brief calculate hilbert key for each body
 *
 * @param[in]  numBodies
 * @param[in]  box      global bounding box, for body coordinate normalization
 * @param[in]  bodyPos  bodies
 * @param[out] keys     output hilbert keys
 * @param[out] values   iota sequence 0...numBodies
 */
__global__ void getKeys(const int numBodies, const Box box, const fvec4* bodyPos, uint64_t* keys, int* values)
{
    const int bodyIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bodyIdx >= numBodies) return;
    const fvec3 pos      = make_fvec3(bodyPos[bodyIdx]);
    const float diameter = 2 * box.R / (1 << NBITS);
    const fvec3 Xmin     = box.X - box.R;
    const fvec3 iX       = (pos - Xmin) / diameter;
    keys[bodyIdx]        = getHilbert(make_int3(iX[0], iX[1], iX[2]));
    values[bodyIdx]      = bodyIdx;
}

//! @brief gather bodyPos into bodyPos2 through new order given by @p value
__global__ void permuteBodies(const int numBodies, const int* value, const fvec4* bodyPos, fvec4* bodyPos2)
{
    const int bodyIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (bodyIdx >= numBodies) return;
    bodyPos2[bodyIdx] = bodyPos[value[bodyIdx]];
}

/*! @brief determine which body indices mark start and ends of <levelrange> nodes
 *
 * launch config: one thread per body
 *
 * @param[in]  numBodies   number of bodies, length of the input arrays @p keys @p keys2 @p bodyBegin @p bodyEnd
 * @param[in]  mask        one-bits set for the first <levelrange> levels
 * @param[in]  keys        sorted body hilbert keys
 * @param[out] keys2       reflected & masked keys
 * @param[out] bodyBegin   bodyBegin[i] = i if keys[i] is first body in a <levelrange> node, 0 otherwise
 * @param[out] bodyEnd     bodyEnd[numBodies - 1 - i] = i + 1 if keys[i] is the last body in a <levelrange> node,
 *                         0 otherwise
 */
__global__ void maskKeys(const int numBodies, const uint64_t mask, const uint64_t* keys, uint64_t* keys2, int* bodyBegin,
                         int* bodyEnd)
{
    const int bodyIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bodyIdx >= numBodies) return;

    keys2[numBodies - bodyIdx - 1] = keys[bodyIdx] & mask;
    const int nextIdx              = min(bodyIdx + 1, numBodies - 1);
    const int prevIdx              = max(bodyIdx - 1, 0);
    const uint64_t currKey         = keys[bodyIdx] & mask;
    const uint64_t nextKey         = keys[nextIdx] & mask;
    const uint64_t prevKey         = keys[prevIdx] & mask;
    if (prevKey < currKey || bodyIdx == 0)
        bodyBegin[bodyIdx] = bodyIdx;
    else
        bodyBegin[bodyIdx] = 0;
    if (currKey < nextKey || bodyIdx == numBodies - 1)
        bodyEnd[numBodies - 1 - bodyIdx] = bodyIdx + 1;
    else
        bodyEnd[numBodies - 1 - bodyIdx] = 0;
}

/*! @brief calculate target ranges
 *
 * launch config: one thread per body
 *
 * @param[in]  numBodies         number of bodies
 * @param[in]  bodyBeginGlob
 * @param[in]  bodyEndGlob
 * @param[out] targetRange
 *
 * A target is a group of consecutive bodies in the same tree cell of at most WARP_SIZE * 2 bodies
 *
 */
__global__ void getTargetRange(const int numBodies, const int* bodyBeginGlob, const int* bodyEndGlob, int2* targetRange)
{
    const int groupSize = WARP_SIZE * 2;
    const int bodyIdx   = blockDim.x * blockIdx.x + threadIdx.x;
    if (bodyIdx >= numBodies) return;

    const int bodyBegin = bodyBeginGlob[bodyIdx];
    assert(bodyIdx >= bodyBegin);

    //! if there are more than groupSize particles in the tree cell, the cell is split into several groups
    //! groupIdx is
    const int groupIdx = (bodyIdx - bodyBegin) / groupSize;

    const int groupBegin = bodyBegin + groupIdx * groupSize;

    if (bodyIdx == groupBegin)
    {
        const int targetIdx    = atomicAdd(&numTargetGlob, 1);
        const int bodyEnd      = bodyEndGlob[numBodies - 1 - bodyIdx];
        targetRange[targetIdx] = make_int2(groupBegin, min(groupSize, bodyEnd - groupBegin));
    }
}
} // namespace

class Group
{
public:
    /*! @brief find groups of up to 64 key-consecutive bodies that are contained within a level-@p levelSplit cell
     *
     * @param[in]  bodyPos       input bodies
     * @param[out] bodyPos2      key-sorted bodies
     * @param[in]  box           coordinate bounding box
     * @param[out] targetRange   (index, count) pairs, containing the first body of the group in bodyPos2, i.e.
     *                           the sorted body array and the number of bodies in the group
     * @param[in]  levelSplit    the number of levels that the bodies in each group are guaranteed to have in common,
     *                           i.e with 3*levelSplit bits matching in the SFC keys
     * @return                   the number of groups
     */
    int targets(cudaVec<fvec4>& bodyPos, cudaVec<fvec4>& bodyPos2, Box box, cudaVec<int2>& targetRange, int levelSplit)
    {
        const int numBodies = bodyPos.size();
        const int NBLOCK    = (numBodies - 1) / NTHREAD + 1;
        cudaVec<uint64_t> key(numBodies);
        cudaVec<int> value(numBodies);
        cudaDeviceSynchronize();

        auto t0 = std::chrono::high_resolution_clock::now();

        //! calculate hilbert key for each particle
        getKeys<<<NBLOCK, NTHREAD>>>(numBodies, box, bodyPos.d(), key.d(), value.d());

        //! sort_by_key
        sort(numBodies, key.d(), value.d());
        //! reorder bodies into hilbert order from bodyPos into bodyPos2
        permuteBodies<<<NBLOCK, NTHREAD>>>(numBodies, value.d(), bodyPos.d(), bodyPos2.d());

        cudaVec<int> bodyBegin(numBodies);
        cudaVec<int> bodyEnd(numBodies);
        cudaVec<uint64_t> key2(numBodies);

        uint64_t mask = 0;
        for (int i = 0; i < NBITS; i++)
        {
            mask <<= 3;
            if (i < levelSplit) mask |= 0x7;
        }

        maskKeys<<<NBLOCK, NTHREAD>>>(numBodies, mask, key.d(), key2.d(), bodyBegin.d(), bodyEnd.d());
        scan(numBodies, key.d(), bodyBegin.d());
        scan(numBodies, key2.d(), bodyEnd.d());

        resetNumTargets<<<1,1>>>();
        getTargetRange<<<NBLOCK, NTHREAD>>>(numBodies, bodyBegin.d(), bodyEnd.d(), targetRange.d());
        kernelSuccess("groupTargets");

        auto t1   = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();

        fprintf(stdout, "Make groups          : %.7f s\n", dt);

        int numTargets;
        CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&numTargets, numTargetGlob, sizeof(int)));

        return numTargets;
    }
};

