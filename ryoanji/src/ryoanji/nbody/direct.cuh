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

#include "kahan.hpp"
#include "kernel.hpp"
#include "types.h"

namespace ryoanji
{

template<class T>
using kvec4 = util::array<kahan<T>, 4>;

struct DirectConfig
{
    static constexpr int numThreads = 256;
};

template<class T>
__global__ void directKernel(int numSource, const T* __restrict__ x, const T* __restrict__ y, const T* __restrict__ z,
                             const T* __restrict__ m, const T* __restrict__ h, T* p, T* ax, T* ay, T* az)
{
    unsigned targetIdx = blockDim.x * blockIdx.x + threadIdx.x;

    Vec3<T> pos_i = {T(0), T(0), T(0)};
    T       h_i   = 1.0;
    if (targetIdx < numSource)
    {
        pos_i = {x[targetIdx], y[targetIdx], z[targetIdx]};
        h_i   = h[targetIdx];
    }

    // kvec4<T> acc = {0.0, 0.0, 0.0, 0.0};
    util::array<T, 4> acc{0, 0, 0, 0};

    __shared__ util::array<T, 5> sm_bodytile[DirectConfig::numThreads];
    for (int tile = 0; tile < gridDim.x; ++tile)
    {
        int sourceIdx = tile * blockDim.x + threadIdx.x;
        if (sourceIdx < numSource)
            sm_bodytile[threadIdx.x] = {x[sourceIdx], y[sourceIdx], z[sourceIdx], m[sourceIdx], h[sourceIdx]};
        else
            sm_bodytile[threadIdx.x] = {0, 0, 0, 0, 1.0};

        __syncthreads();

        for (int j = 0; j < blockDim.x; ++j)
        {
            Vec3<T> pos_j{sm_bodytile[j][0], sm_bodytile[j][1], sm_bodytile[j][2]};
            T       q_j = sm_bodytile[j][3];
            T       h_j = sm_bodytile[j][4];

            acc = P2P(acc, pos_i, pos_j, q_j, h_i, h_j);
        }

        __syncthreads();
    }

    if (targetIdx < numSource)
    {
        p[targetIdx]  = m[targetIdx] * T(acc[0]);
        ax[targetIdx] = T(acc[1]);
        ay[targetIdx] = T(acc[2]);
        az[targetIdx] = T(acc[3]);
    }
}

template<class T>
void directSum(std::size_t numBodies, const T* x, const T* y, const T* z, const T* m, const T* h, T* p, T* ax, T* ay,
               T* az)
{
    int numThreads = DirectConfig::numThreads;
    int numBlock   = (numBodies - 1) / numThreads + 1;

    directKernel<<<numBlock, numThreads>>>(numBodies, x, y, z, m, h, p, ax, ay, az);
    kernelSuccess("direct sum");
}

} // namespace ryoanji