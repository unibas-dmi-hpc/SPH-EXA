#pragma once

#include <chrono>

#define NBITS 21

extern void sort(const int size, uint64_t* key, int* value);
extern void scan(const int size, uint64_t* key, int* value);

namespace
{
__device__ unsigned int numTargetGlob = 0;

__device__ void swap(int& a, int& b)
{
    int c(a);
    a = b;
    b = c;
};

__device__ uint64_t getHilbert(int3 iX)
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

__global__ void permuteBodies(const int numBodies, const int* value, const fvec4* bodyPos, fvec4* bodyPos2)
{
    const int bodyIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (bodyIdx >= numBodies) return;
    bodyPos2[bodyIdx] = bodyPos[value[bodyIdx]];
}

__global__ void maskKeys(const int numBodies, const uint64_t mask, uint64_t* keys, uint64_t* keys2, int* bodyBegin,
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

__global__ void getTargetRange(const int numBodies, const int* bodyBeginGlob, const int* bodyEndGlob, int2* targetRange)
{
    const int groupSize = WARP_SIZE * 2;
    const int bodyIdx   = blockDim.x * blockIdx.x + threadIdx.x;
    if (bodyIdx >= numBodies) return;
    const int bodyBegin = bodyBeginGlob[bodyIdx];
    assert(bodyIdx >= bodyBegin);
    const int groupIdx   = (bodyIdx - bodyBegin) / groupSize;
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
    int targets(cudaVec<fvec4>& bodyPos, cudaVec<fvec4>& bodyPos2, Box box, cudaVec<int2>& targetRange, int levelSplit)
    {
        const int numBodies = bodyPos.size();
        const int NBLOCK    = (numBodies - 1) / NTHREAD + 1;
        cudaVec<uint64_t> key(numBodies);
        cudaVec<int> value(numBodies);
        cudaDeviceSynchronize();

        auto t0 = std::chrono::high_resolution_clock::now();

        getKeys<<<NBLOCK, NTHREAD>>>(numBodies, box, bodyPos.d(), key.d(), value.d());

        sort(numBodies, key.d(), value.d());
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
