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
 * @brief Benchmark cornerstone octree generation on the GPU
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <chrono>
#include <iostream>
#include <random>

#include <thrust/device_vector.h>

#include "cstone/sfc/hilbert.hpp"

using namespace cstone;

template<class KeyType>
__global__ void
computeHilbertKeysKernel(KeyType* keys,
                         const unsigned* x,
                         const unsigned* y,
                         const unsigned* z,
                         size_t numKeys)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numKeys)
    {
        keys[tid] = iHilbert<KeyType>(x[tid], y[tid], z[tid]);
    }
}

template<class KeyType>
inline void computeHilbertKeys(KeyType* keys,
                               const unsigned* x,
                               const unsigned* y,
                               const unsigned* z,
                               size_t numKeys)
{
    constexpr int threadsPerBlock = 256;
    computeHilbertKeysKernel<<<iceil(numKeys, threadsPerBlock), threadsPerBlock>>>(keys, x, y, z, numKeys);
}

template<class KeyType>
__global__ void
computeMortonKeysKernel(KeyType* keys,
                        const unsigned* x,
                        const unsigned* y,
                        const unsigned* z,
                        size_t numKeys)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numKeys)
    {
        keys[tid] = imorton3D<KeyType>(x[tid], y[tid], z[tid]);
    }
}

template<class KeyType>
inline void computeMortonKeys(KeyType* keys,
                              const unsigned* x,
                              const unsigned* y,
                              const unsigned* z,
                              size_t numKeys)
{
    constexpr int threadsPerBlock = 256;
    computeMortonKeysKernel<<<iceil(numKeys, threadsPerBlock), threadsPerBlock>>>(keys, x, y, z, numKeys);
}

int main()
{
    using KeyType = uint64_t;
    unsigned numKeys = 32000000;

    int maxCoord = (1 << maxTreeLevel<KeyType>{}) - 1;
    std::mt19937 gen;
    std::uniform_int_distribution<unsigned> distribution(0, maxCoord);
    auto getRand = [&distribution, &gen](){ return distribution(gen); };

    std::vector<unsigned> x(numKeys);
    std::vector<unsigned> y(numKeys);
    std::vector<unsigned> z(numKeys);

    std::generate(begin(x), end(x), getRand);
    std::generate(begin(y), end(y), getRand);
    std::generate(begin(z), end(z), getRand);

    thrust::device_vector<unsigned> dx = x;
    thrust::device_vector<unsigned> dy = y;
    thrust::device_vector<unsigned> dz = z;

    thrust::device_vector<KeyType> sfcKeys(numKeys);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, cudaStreamDefault);

    computeHilbertKeys(thrust::raw_pointer_cast(sfcKeys.data()),
                       thrust::raw_pointer_cast(dx.data()),
                       thrust::raw_pointer_cast(dy.data()),
                       thrust::raw_pointer_cast(dz.data()),
                       numKeys);

    cudaEventRecord(stop, cudaStreamDefault);
    cudaEventSynchronize(stop);

    float t0;
    cudaEventElapsedTime(&t0, start, stop);
    std::cout << "compute time for " << numKeys << " hilbert keys: " << t0/1000 << " s" << std::endl;

    cudaEventRecord(start, cudaStreamDefault);

    computeMortonKeys(thrust::raw_pointer_cast(sfcKeys.data()),
                      thrust::raw_pointer_cast(dx.data()),
                      thrust::raw_pointer_cast(dy.data()),
                      thrust::raw_pointer_cast(dz.data()),
                      numKeys);

    cudaEventRecord(stop, cudaStreamDefault);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&t0, start, stop);
    std::cout << "compute time for " << numKeys << " morton keys: " << t0/1000 << " s" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
