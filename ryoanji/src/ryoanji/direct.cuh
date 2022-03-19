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
 * @brief All-to-all direct sum implementation of gravitational forces
 *
 * Inspired by
 * https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "types.h"
#include "kahan.hpp"

namespace ryoanji
{

template<class T>
using kvec4 = util::array<kahan<T>, 4>;

struct DirectConfig
{
    static constexpr int numThreads = 256;
};

template<class T>
__global__ void directKernel(int numSource, T eps2, const T* __restrict__ x, const T* __restrict__ y,
                             const T* __restrict__ z, const T* __restrict__ m, T* p, T* ax, T* ay, T* az)
{
    unsigned targetIdx = blockDim.x * blockIdx.x + threadIdx.x;

    Vec3<T> pos_i = {T(0), T(0), T(0)};
    if (targetIdx < numSource) { pos_i = {x[targetIdx], y[targetIdx], z[targetIdx]}; }

    // kvec4<T> acc = {0.0, 0.0, 0.0, 0.0};
    util::array<double, 4> acc{0, 0, 0, 0};

    __shared__ Vec4<T> sm_bodytile[DirectConfig::numThreads];
    for (int tile = 0; tile < gridDim.x; ++tile)
    {
        int sourceIdx = tile * blockDim.x + threadIdx.x;
        if (sourceIdx < numSource)
            sm_bodytile[threadIdx.x] = {x[sourceIdx], y[sourceIdx], z[sourceIdx], m[sourceIdx]};
        else
            sm_bodytile[threadIdx.x] = Vec4<T>{0, 0, 0, 0};

        __syncthreads();

        for (int j = 0; j < blockDim.x; ++j)
        {
            Vec3<T> pos_j{sm_bodytile[j][0], sm_bodytile[j][1], sm_bodytile[j][2]};
            T       q_j = sm_bodytile[j][3];
            Vec3<T> dX  = pos_j - pos_i;

            T R2    = norm2(dX);
            T invR  = rsqrtf(R2 + eps2);
            T invR2 = invR * invR;
            T invR1 = q_j * invR;

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
        p[targetIdx]  = T(acc[0]);
        ax[targetIdx] = T(acc[1]);
        ay[targetIdx] = T(acc[2]);
        az[targetIdx] = T(acc[3]);
    }
}

template<class T>
void directSum(std::size_t numBodies, const T* x, const T* y, const T* z, const T* m, T* p, T* ax, T* ay, T* az, T eps)
{
    int numThreads = DirectConfig::numThreads;
    int numBlock   = (numBodies - 1) / numThreads + 1;

    directKernel<<<numBlock, numThreads>>>(numBodies, eps * eps, x, y, z, m, p, ax, ay, az);
    ryoanji::kernelSuccess("direct sum");
}

} // namespace ryoanji