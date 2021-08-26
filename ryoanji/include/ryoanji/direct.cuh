#pragma once
#include <algorithm>

#include "types.h"

__global__ void directKernel(int numSource, float eps2, fvec4* bodyAcc)
{
    unsigned laneIdx      = threadIdx.x & (WARP_SIZE - 1);
    // number of warps in the block
    unsigned numWarps    = (blockDim.x - 1) / WARP_SIZE + 1;
    unsigned targetIdx   = blockDim.x * blockIdx.x + threadIdx.x;

    fvec4 pos = tex1Dfetch(texBody, targetIdx);

    const fvec3 pos_i(pos[0], pos[1], pos[2]);
    kvec4 acc = {0.0, 0.0, 0.0, 0.0};

    for (int block = 0; block < gridDim.x; ++block)
    {
        unsigned blockOffset = block * blockDim.x;
        for (int jb = 0; jb < numWarps; jb++)
        {
            pos[3] = 0;
            unsigned sourceIdx = blockOffset + jb * WARP_SIZE + laneIdx;

            if (sourceIdx < numSource)
            {
                pos = tex1Dfetch(texBody, sourceIdx);
            }

            for (int j = 0; j < WARP_SIZE; j++)
            {
                fvec3 pos_j(__shfl_sync(0xFFFFFFFF, pos[0], j),
                            __shfl_sync(0xFFFFFFFF, pos[1], j),
                            __shfl_sync(0xFFFFFFFF, pos[2], j));

                float q_j = __shfl_sync(0xFFFFFFFF, pos[3], j);
                fvec3 dX = pos_j - pos_i;

                float R2    = norm(dX) + eps2;
                float invR  = rsqrtf(R2);
                float invR2 = invR * invR;
                float invR1 = q_j * invR;

                dX *= invR1 * invR2;

                acc[0] -= invR1;
                acc[1] += dX[0];
                acc[2] += dX[1];
                acc[3] += dX[2];
            }
        }
    }

    if (targetIdx < numSource)
    {
        bodyAcc[targetIdx] = fvec4(acc[0], acc[1], acc[2], acc[3]);
    }
}

void directSum(float eps, cudaVec<fvec4>& bodyPos, cudaVec<fvec4>& bodyAcc)
{
    int numBodies  = bodyPos.size();
    int numThreads = std::min(512, numBodies);
    int numBlock   = (numBodies - 1) / numThreads + 1;

    bodyPos.bind(texBody);
    directKernel<<<numBlock, numThreads>>>(numBodies, eps * eps, bodyAcc.d());
    bodyPos.unbind(texBody);
    cudaDeviceSynchronize();
}
