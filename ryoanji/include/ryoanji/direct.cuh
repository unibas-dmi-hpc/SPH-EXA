#pragma once

#include "types.h"
#include "kahan.hpp"

typedef util::array<kahan<float>, 4> kvec4;

struct DirectConfig
{
    static constexpr int numThreads = 256;
};

__global__ void directKernel(int numSource, float eps2, const fvec4* __restrict__ bodyPos, fvec4* bodyAcc)
{
    unsigned targetIdx = blockDim.x * blockIdx.x + threadIdx.x;

    fvec4 pos = {0.0, 0.0, 0.0, 0.0};
    if (targetIdx < numSource)
    {
        pos = bodyPos[targetIdx];
    }
    const fvec3 pos_i{pos[0], pos[1], pos[2]};

    //kvec4 acc = {0.0, 0.0, 0.0, 0.0};
    util::array<double, 4> acc{0, 0, 0, 0};

    __shared__ fvec4 sm_bodytile[DirectConfig::numThreads];
    for (int tile = 0; tile < gridDim.x; ++tile)
    {
        int sourceIdx = tile * blockDim.x + threadIdx.x;
        if (sourceIdx < numSource)
            sm_bodytile[threadIdx.x] = bodyPos[sourceIdx];
        else
            sm_bodytile[threadIdx.x] = fvec4{0, 0, 0, 0};

        __syncthreads();

        for (int j = 0; j < blockDim.x; ++j)
        {
            fvec3 pos_j{sm_bodytile[j][0], sm_bodytile[j][1], sm_bodytile[j][2]};
            float q_j = sm_bodytile[j][3];
            fvec3 dX  = pos_j - pos_i;

            float R2    = norm2(dX);
            float invR  = rsqrtf(R2 + eps2);
            float invR2 = invR * invR;
            float invR1 = q_j * invR;

            dX *= invR1 * invR2;

            // avoid self gravity
            if (R2 != 0.0f)
            {
                acc[0] -= invR1;
                acc[1] += dX[0];
                acc[2] += dX[1];
                acc[3] += dX[2];
            }
        }

        __syncthreads();
    }

    if (targetIdx < numSource)
    {
        bodyAcc[targetIdx] = fvec4{float(acc[0]), float(acc[1]), float(acc[2]), float(acc[3])};
    }
}

void directSum(std::size_t numBodies, const fvec4* bodyPos, fvec4* bodyAcc, float eps)
{
    int numThreads = DirectConfig::numThreads;
    int numBlock   = (numBodies - 1) / numThreads + 1;

    directKernel<<<numBlock, numThreads>>>(numBodies, eps * eps, bodyPos, bodyAcc);
    cudaDeviceSynchronize();
}

