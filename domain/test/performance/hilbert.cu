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

#include <iostream>
#include <random>

#include <thrust/device_vector.h>

#include "cstone/sfc/sfc.hpp"
#include "cstone/util/util.hpp"

using namespace cstone;

template<class KeyType>
__global__ void
computeSfcKeysKernel(KeyType* keys,
                     const unsigned* x,
                     const unsigned* y,
                     const unsigned* z,
                     size_t numKeys)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numKeys)
    {
        keys[tid] = iSfcKey<KeyType>(x[tid], y[tid], z[tid]);
    }
}

template<class KeyType>
inline void computeSfcKeys(KeyType* keys,
                           const unsigned* x,
                           const unsigned* y,
                           const unsigned* z,
                           size_t numKeys)
{
    constexpr int threadsPerBlock = 256;
    computeSfcKeysKernel<<<iceil(numKeys, threadsPerBlock), threadsPerBlock>>>(keys, x, y, z, numKeys);
}

template<class F>
float timeGpu(F&& f)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, cudaStreamDefault);

    f();

    cudaEventRecord(stop, cudaStreamDefault);
    cudaEventSynchronize(stop);

    float t0;
    cudaEventElapsedTime(&t0, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return t0;
}

int main()
{
    using IntegerType = uint64_t;
    unsigned numKeys = 32000000;

    unsigned maxCoord = (1 << maxTreeLevel<IntegerType>{}) - 1;
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

    thrust::device_vector<MortonKey<IntegerType>>  mortonKeys(numKeys);
    thrust::device_vector<HilbertKey<IntegerType>> hilbertKeys(numKeys);

    auto computeHilbert = [&]()
    {
        computeSfcKeys(thrust::raw_pointer_cast(hilbertKeys.data()), thrust::raw_pointer_cast(dx.data()),
                       thrust::raw_pointer_cast(dy.data()), thrust::raw_pointer_cast(dz.data()), numKeys);
    };

    auto computeMorton = [&]()
    {
        computeSfcKeys(thrust::raw_pointer_cast(mortonKeys.data()), thrust::raw_pointer_cast(dx.data()),
                       thrust::raw_pointer_cast(dy.data()), thrust::raw_pointer_cast(dz.data()), numKeys);
    };

    float t_hilbert = timeGpu(computeHilbert);
    float t_morton = timeGpu(computeMorton);
    std::cout << "compute time for " << numKeys << " hilbert keys: " << t_hilbert/1000 << " s" << std::endl;
    std::cout << "compute time for " << numKeys << " morton keys: " << t_morton/1000 << " s" << std::endl;
}
