#pragma once
#include <algorithm>

#include "types.h"

__global__ void directKernel(const int numSource, const int images, const float EPS2, const float cycle, fvec4* bodyAcc)
{
    const int laneIdx      = threadIdx.x & (WARP_SIZE - 1);
    const int numChunk     = (numSource - 1) / gridDim.x + 1;
    const int numWarpChunk = (numChunk - 1) / WARP_SIZE + 1;
    const int blockOffset  = blockIdx.x * numChunk;
    fvec4 pos              = tex1Dfetch(texBody, threadIdx.x);

    const fvec3 pos_i(pos[0], pos[1], pos[2]);
    kvec4 acc = {0.0, 0.0, 0.0, 0.0};
    fvec3 Xperiodic = 0.0f;

    for (int jb = 0; jb < numWarpChunk; jb++)
    {
        const int sourceIdx = min(blockOffset + jb * WARP_SIZE + laneIdx, numSource - 1);
        pos                 = tex1Dfetch(texBody, sourceIdx);
        if (sourceIdx >= numSource) pos[3] = 0;
        for (int ix = -images; ix <= images; ix++)
        {
            for (int iy = -images; iy <= images; iy++)
            {
                for (int iz = -images; iz <= images; iz++)
                {
                    Xperiodic[0] = ix * cycle;
                    Xperiodic[1] = iy * cycle;
                    Xperiodic[2] = iz * cycle;
                    for (int j = 0; j < WARP_SIZE; j++)
                    {
                        const fvec3 pos_j(__shfl(pos[0], j), __shfl(pos[1], j), __shfl(pos[2], j));
                        const float q_j   = __shfl(pos[3], j);
                        fvec3 dX          = pos_j - pos_i - Xperiodic;
                        const float R2    = norm(dX) + EPS2;
                        const float invR  = rsqrtf(R2);
                        const float invR2 = invR * invR;
                        const float invR1 = q_j * invR;
                        dX *= invR1 * invR2;
                        acc[0] -= invR1;
                        acc[1] += dX[0];
                        acc[2] += dX[1];
                        acc[3] += dX[2];
                    }
                }
            }
        }
    }

    unsigned targetIdx = blockIdx.x * blockDim.x + threadIdx.x;
    bodyAcc[targetIdx] = fvec4(acc[0], acc[1], acc[2], acc[3]);
}

void directSum(const int numTarget, const int numBlock, const int images, const float eps, const float cycle,
               cudaVec<fvec4>& bodyPos2, cudaVec<fvec4>& bodyAcc2)
{
    const int numBodies = bodyPos2.size();
    bodyPos2.bind(texBody);
    directKernel<<<numBlock, numTarget>>>(numBodies, images, eps * eps, cycle, bodyAcc2.d());
    bodyPos2.unbind(texBody);
    cudaDeviceSynchronize();
}
