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

#include "timing.cuh"

using namespace cstone;

template<class KeyType>
__global__ void
computeSfcKeysKernel(KeyType* keys, const unsigned* x, const unsigned* y, const unsigned* z, size_t numKeys)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numKeys)
    {
        keys[tid] = iSfcKey<KeyType>(x[tid], y[tid], z[tid]);
    }
}

template<class KeyType>
inline void computeSfcKeys(KeyType* keys, const unsigned* x, const unsigned* y, const unsigned* z, size_t numKeys)
{
    constexpr int threadsPerBlock = 256;
    computeSfcKeysKernel<<<iceil(numKeys, threadsPerBlock), threadsPerBlock>>>(keys, x, y, z, numKeys);
}

template<class KeyType, class T>
__global__ void
computeSfcKeysRealKernel(KeyType* keys, const T* x, const T* y, const T* z, size_t numKeys, const Box<T> box)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numKeys)
    {
        keys[tid] = sfc3D<KeyType>(x[tid], y[tid], z[tid], box);
    }
}

template<class KeyType, class T>
inline void computeSfcRealKeys(KeyType* keys, const T* x, const T* y, const T* z, size_t numKeys, const Box<T>& box)
{
    constexpr int threadsPerBlock = 256;
    computeSfcKeysRealKernel<<<iceil(numKeys, threadsPerBlock), threadsPerBlock>>>(keys, x, y, z, numKeys, box);
}

template<class KeyType>
__global__ void decodeSfcKeysKernel(const KeyType* keys, unsigned* x, unsigned* y, unsigned* z, size_t numKeys)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numKeys)
    {
        thrust::tie(x[tid], y[tid], z[tid]) = decodeSfc(keys[tid]);
    }
}

template<class KeyType>
inline void decodeSfcKeys(const KeyType* keys, unsigned* x, unsigned* y, unsigned* z, size_t numKeys)
{
    constexpr int threadsPerBlock = 256;
    decodeSfcKeysKernel<<<iceil(numKeys, threadsPerBlock), threadsPerBlock>>>(keys, x, y, z, numKeys);
}

int main()
{
    using IntegerType = uint64_t;
    unsigned numKeys  = 32000000;

    using Real = double;
    Box<Real> box(-1, 1);

    std::mt19937 gen;
    std::uniform_real_distribution<Real> distribution(box.xmin(), box.xmax());
    auto getRand = [&distribution, &gen]() { return distribution(gen); };

    std::vector<Real> x(numKeys);
    std::vector<Real> y(numKeys);
    std::vector<Real> z(numKeys);

    std::generate(begin(x), end(x), getRand);
    std::generate(begin(y), end(y), getRand);
    std::generate(begin(z), end(z), getRand);

    thrust::device_vector<MortonKey<IntegerType>>  mortonKeys(numKeys);
    thrust::device_vector<HilbertKey<IntegerType>> hilbertKeys(numKeys);

    {
        std::vector<unsigned> ix(numKeys);
        std::vector<unsigned> iy(numKeys);
        std::vector<unsigned> iz(numKeys);

        auto normIntX = [&box](Real a) { return toNBitInt<IntegerType>(normalize(a, box.xmin(), box.xmax())); };
        auto normIntY = [&box](Real a) { return toNBitInt<IntegerType>(normalize(a, box.ymin(), box.ymax())); };
        auto normIntZ = [&box](Real a) { return toNBitInt<IntegerType>(normalize(a, box.zmin(), box.zmax())); };
        std::transform(begin(x), end(x), begin(ix), normIntX);
        std::transform(begin(y), end(y), begin(iy), normIntY);
        std::transform(begin(z), end(z), begin(iz), normIntZ);

        thrust::device_vector<unsigned> dx = ix;
        thrust::device_vector<unsigned> dy = iy;
        thrust::device_vector<unsigned> dz = iz;

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
        std::cout << "compute time for " << numKeys << " hilbert keys: " << t_hilbert / 1000 << " s" << std::endl;
        std::cout << "compute time for " << numKeys << " morton keys: " << t_morton / 1000 << " s" << std::endl;

        thrust::device_vector<unsigned> dx2(numKeys);
        thrust::device_vector<unsigned> dy2(numKeys);
        thrust::device_vector<unsigned> dz2(numKeys);

        auto decodeHilbert = [&]()
        {
            decodeSfcKeys(thrust::raw_pointer_cast(hilbertKeys.data()), thrust::raw_pointer_cast(dx2.data()),
                          thrust::raw_pointer_cast(dy2.data()), thrust::raw_pointer_cast(dz2.data()), numKeys);
        };

        float t_decode = timeGpu(decodeHilbert);
        bool passDecode = thrust::equal(dx.begin(), dx.end(), dx2.begin()) &&
                          thrust::equal(dy.begin(), dy.end(), dy2.begin()) &&
                          thrust::equal(dz.begin(), dz.end(), dz2.begin());
        std::string result = (passDecode) ? "pass" : "fail";
        std::cout << "decode time for " << numKeys << " hilbert keys: " << t_decode / 1000 << " s, result: " << result
                  << std::endl;
    }

    thrust::device_vector<MortonKey<IntegerType>>  mortonKeys2(numKeys);
    thrust::device_vector<HilbertKey<IntegerType>> hilbertKeys2(numKeys);

    {
        thrust::device_vector<Real> dx = x;
        thrust::device_vector<Real> dy = y;
        thrust::device_vector<Real> dz = z;

        auto computeHilbert = [&]()
        {
            computeSfcRealKeys(thrust::raw_pointer_cast(hilbertKeys2.data()), thrust::raw_pointer_cast(dx.data()),
                               thrust::raw_pointer_cast(dy.data()), thrust::raw_pointer_cast(dz.data()), numKeys, box);
        };

        auto computeMorton = [&]()
        {
            computeSfcRealKeys(thrust::raw_pointer_cast(mortonKeys2.data()), thrust::raw_pointer_cast(dx.data()),
                               thrust::raw_pointer_cast(dy.data()), thrust::raw_pointer_cast(dz.data()), numKeys, box);
        };

        float t_hilbert = timeGpu(computeHilbert);
        float t_morton  = timeGpu(computeMorton);
        std::cout << "compute time for " << numKeys << " hilbert keys from doubles : "
                  << t_hilbert / 1000 << " s" << std::endl;
        std::cout << "compute time for " << numKeys << " morton keys from doubles: "
                  << t_morton / 1000 << " s" << std::endl;
    }

    std::cout << "keys match: " << thrust::equal(hilbertKeys.begin(), hilbertKeys.end(), hilbertKeys2.begin())
              << std::endl;
}
