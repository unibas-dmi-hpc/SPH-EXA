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
 * @brief  Basic algorithms on the GPU
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <cassert>

#include <cub/cub.cuh>

#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "cstone/cuda/errorcheck.cuh"
#include "cstone/primitives/math.hpp"
#include "cstone/util/array.hpp"
#include "primitives_gpu.h"

namespace cstone
{

template<class T>
void fillGpu(T* first, T* last, T value)
{
    thrust::fill(thrust::device, first, last, value);
}

template void fillGpu(double*, double*, double);
template void fillGpu(float*, float*, float);
template void fillGpu(int*, int*, int);
template void fillGpu(char*, char*, char);
template void fillGpu(unsigned*, unsigned*, unsigned);
template void fillGpu(uint64_t*, uint64_t*, uint64_t);

template<class T>
struct ScaleFunctor
{
    const T s;

    ScaleFunctor(T s_)
        : s(s_)
    {
    }

    __host__ __device__ T operator()(const T& x) const { return s * x; }
};

template<class T>
void scaleGpu(T* first, T* last, T value)
{
    thrust::transform(thrust::device, first, last, first, ScaleFunctor<T>(value));
}

template void scaleGpu(double*, double*, double);
template void scaleGpu(float*, float*, float);

template<class T>
struct IncrementFunctor
{
    const T s;

    IncrementFunctor(T s_)
        : s(s_)
    {
    }

    __host__ __device__ T operator()(const T& x) const { return x + s; }
};

template<class T>
void incrementGpu(const T* first, const T* last, T* d_first, T value)
{
    thrust::transform(thrust::device, first, last, d_first, IncrementFunctor<T>(value));
}

#define INCREMENT_GPU(T) template void incrementGpu(const T* first, const T* last, T* d_first, T value)
INCREMENT_GPU(unsigned);
INCREMENT_GPU(uint64_t);

template<class T, class IndexType>
__global__ void gatherGpuKernel(const IndexType* map, size_t n, const T* source, T* destination)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) { destination[tid] = source[map[tid]]; }
}

template<class T, class IndexType>
void gatherGpu(const IndexType* map, size_t n, const T* source, T* destination)
{
    int numThreads = 256;
    int numBlocks  = iceil(n, numThreads);

    gatherGpuKernel<<<numBlocks, numThreads>>>(map, n, source, destination);
}

template void gatherGpu(const int*, size_t, const int*, int*);
template void gatherGpu(const int*, size_t, const uint32_t*, uint32_t*);
template void gatherGpu(const int*, size_t, const uint64_t*, uint64_t*);
template void gatherGpu(const int*, size_t, const util::array<float, 4>*, util::array<float, 4>*);
template void gatherGpu(const int*, size_t, const util::array<float, 8>*, util::array<float, 8>*);
template void gatherGpu(const int*, size_t, const util::array<float, 20>*, util::array<float, 20>*);
template void gatherGpu(const int*, size_t, const util::array<double, 4>*, util::array<double, 4>*);
template void gatherGpu(const int*, size_t, const util::array<double, 8>*, util::array<double, 8>*);
template void gatherGpu(const int*, size_t, const util::array<double, 20>*, util::array<double, 20>*);

template void gatherGpu(const unsigned*, size_t, const uint8_t*, uint8_t*);
template void gatherGpu(const unsigned*, size_t, const double*, double*);
template void gatherGpu(const unsigned*, size_t, const float*, float*);
template void gatherGpu(const unsigned*, size_t, const char*, char*);
template void gatherGpu(const unsigned*, size_t, const int*, int*);
template void gatherGpu(const unsigned*, size_t, const long*, long*);
template void gatherGpu(const unsigned*, size_t, const unsigned*, unsigned*);
template void gatherGpu(const unsigned*, size_t, const unsigned long*, unsigned long*);
template void gatherGpu(const unsigned*, size_t, const unsigned long long*, unsigned long long*);
template void gatherGpu(const unsigned*, size_t, const util::array<float, 1>*, util::array<float, 1>*);
template void gatherGpu(const unsigned*, size_t, const util::array<float, 2>*, util::array<float, 2>*);
template void gatherGpu(const unsigned*, size_t, const util::array<float, 3>*, util::array<float, 3>*);
template void gatherGpu(const unsigned*, size_t, const util::array<float, 4>*, util::array<float, 4>*);

template<class T, class IndexType>
__global__ void scatterGpuKernel(const IndexType* map, size_t n, const T* source, T* destination)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) { destination[map[tid]] = source[tid]; }
}

template<class T, class IndexType>
void scatterGpu(const IndexType* map, size_t n, const T* source, T* destination)
{
    int numThreads = 256;
    int numBlocks  = iceil(n, numThreads);

    scatterGpuKernel<<<numBlocks, numThreads>>>(map, n, source, destination);
}

template void scatterGpu(const int*, size_t, const int*, int*);
template void scatterGpu(const int*, size_t, const uint32_t*, uint32_t*);
template void scatterGpu(const int*, size_t, const uint64_t*, uint64_t*);
template void scatterGpu(const int*, size_t, const util::array<float, 4>*, util::array<float, 4>*);
template void scatterGpu(const int*, size_t, const util::array<float, 8>*, util::array<float, 8>*);
template void scatterGpu(const int*, size_t, const util::array<float, 20>*, util::array<float, 20>*);
template void scatterGpu(const int*, size_t, const util::array<double, 4>*, util::array<double, 4>*);
template void scatterGpu(const int*, size_t, const util::array<double, 8>*, util::array<double, 8>*);
template void scatterGpu(const int*, size_t, const util::array<double, 20>*, util::array<double, 20>*);

template<class T>
std::tuple<T, T> MinMaxGpu<T>::operator()(const T* first, const T* last)
{
    auto minMax = thrust::minmax_element(thrust::device, first, last);

    T theMinimum, theMaximum;
    checkGpuErrors(cudaMemcpy(&theMinimum, minMax.first, sizeof(T), cudaMemcpyDeviceToHost));
    checkGpuErrors(cudaMemcpy(&theMaximum, minMax.second, sizeof(T), cudaMemcpyDeviceToHost));

    return std::make_tuple(theMinimum, theMaximum);
}

template class MinMaxGpu<double>;
template class MinMaxGpu<float>;

using thrust::get;

template<class T>
struct NormSquare3D
{
    HOST_DEVICE_FUN T operator()(const thrust::tuple<T, T, T>& X)
    {
        return get<0>(X) * get<0>(X) + get<1>(X) * get<1>(X) + get<2>(X) * get<2>(X);
    }
};

template<class T>
T maxNormSquareGpu(const T* x, const T* y, const T* z, size_t numElements)
{
    auto it1 = thrust::make_zip_iterator(x, y, z);
    auto it2 = thrust::make_zip_iterator(x + numElements, y + numElements, z + numElements);

    T init = 0;

    return thrust::transform_reduce(thrust::device, it1, it2, NormSquare3D<T>{}, init, thrust::maximum<T>{});
}

template float maxNormSquareGpu(const float*, const float*, const float*, size_t);
template double maxNormSquareGpu(const double*, const double*, const double*, size_t);

template<class T>
size_t lowerBoundGpu(const T* first, const T* last, T value)
{
    return thrust::lower_bound(thrust::device, first, last, value) - first;
}

template size_t lowerBoundGpu(const unsigned*, const unsigned*, unsigned);
template size_t lowerBoundGpu(const uint64_t*, const uint64_t*, uint64_t);
template size_t lowerBoundGpu(const int*, const int*, int);
template size_t lowerBoundGpu(const int64_t*, const int64_t*, int64_t);
template size_t lowerBoundGpu(const float*, const float*, float);

template<class T, class IndexType>
void lowerBoundGpu(const T* first, const T* last, const T* valueFirst, const T* valueLast, IndexType* result)
{
    thrust::lower_bound(thrust::device, first, last, valueFirst, valueLast, result);
}

template void lowerBoundGpu(const unsigned*, const unsigned*, const unsigned*, const unsigned*, unsigned*);
template void lowerBoundGpu(const uint64_t*, const uint64_t*, const uint64_t*, const uint64_t*, unsigned*);
template void lowerBoundGpu(const unsigned*, const unsigned*, const unsigned*, const unsigned*, uint64_t*);
template void lowerBoundGpu(const uint64_t*, const uint64_t*, const uint64_t*, const uint64_t*, uint64_t*);

template<class Tin, class Tout, class IndexType>
__global__ void segmentMaxKernel(const Tin* input, const IndexType* segments, size_t numSegments, Tout* output)
{
    IndexType tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numSegments)
    {
        Tin localMax = 0;

        IndexType segStart = segments[tid];
        IndexType segEnd   = segments[tid + 1];

        for (IndexType i = segStart; i < segEnd; ++i)
        {
            localMax = max(localMax, input[i]);
        }

        output[tid] = Tout(localMax);
    }
}

template<class Tin, class Tout, class IndexType>
void segmentMax(const Tin* input, const IndexType* segments, size_t numSegments, Tout* output)
{
    int numThreads = 256;
    int numBlocks  = iceil(numSegments, numThreads);

    segmentMaxKernel<<<numBlocks, numThreads>>>(input, segments, numSegments, output);
}

template void segmentMax(const float*, const unsigned*, size_t, float*);
template void segmentMax(const double*, const unsigned*, size_t, float*);
template void segmentMax(const double*, const unsigned*, size_t, double*);
template void segmentMax(const float*, const uint64_t*, size_t, float*);
template void segmentMax(const double*, const uint64_t*, size_t, float*);
template void segmentMax(const double*, const uint64_t*, size_t, double*);

template<class Tin, class Tout>
Tout reduceGpu(const Tin* input, size_t numElements, Tout init)
{
    return thrust::reduce(thrust::device, input, input + numElements, init);
}

template size_t reduceGpu(const unsigned*, size_t, size_t);

template<class IndexType>
void sequenceGpu(IndexType* input, size_t numElements, IndexType init)
{
    thrust::sequence(thrust::device, input, input + numElements, init);
}

template void sequenceGpu(int*, size_t, int);
template void sequenceGpu(unsigned*, size_t, unsigned);
template void sequenceGpu(uint64_t*, uint64_t, uint64_t);

template<class KeyType>
void sortGpu(KeyType* first, KeyType* last, KeyType* keyBuf)
{
    size_t numElements = last - first;

    cub::DoubleBuffer<KeyType> d_keys(first, keyBuf);

    // Determine temporary device storage requirements
    void* d_tempStorage     = nullptr;
    size_t tempStorageBytes = 0;
    cub::DeviceRadixSort::SortKeys(d_tempStorage, tempStorageBytes, d_keys, numElements);

    // Allocate temporary storage
    checkGpuErrors(cudaMalloc(&d_tempStorage, tempStorageBytes));

    // Run sorting operation
    checkGpuErrors(cub::DeviceRadixSort::SortKeys(d_tempStorage, tempStorageBytes, d_keys, numElements));

    auto* curValues = d_keys.Current();
    if (curValues != first)
    {
        checkGpuErrors(cudaMemcpy(first, curValues, numElements * sizeof(KeyType), cudaMemcpyDeviceToDevice));
    }

    checkGpuErrors(cudaFree(d_tempStorage));
}

template void sortGpu(uint32_t*, uint32_t*, uint32_t*);
template void sortGpu(uint64_t*, uint64_t*, uint64_t*);
template void sortGpu(float*, float*, float*);

// Determine temporary device storage requirements
template<class KeyType, class ValueType>
uint64_t sortByKeyTempStorage(uint64_t numElements)
{
    cub::DoubleBuffer<KeyType> d_keys(nullptr, nullptr);
    cub::DoubleBuffer<ValueType> d_values(nullptr, nullptr);

    uint64_t tempStorageBytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, tempStorageBytes, d_keys, d_values, numElements);
    return tempStorageBytes;
}

template<class KeyType, class ValueType>
void sortByKeyGpu(KeyType* first,
                  KeyType* last,
                  ValueType* values,
                  KeyType* keyBuf,
                  ValueType* valueBuf,
                  void* d_tempStorage,
                  uint64_t tempStorageBytes)
{
    size_t numElements = last - first;

    cub::DoubleBuffer<KeyType> d_keys(first, keyBuf);
    cub::DoubleBuffer<ValueType> d_values(values, valueBuf);

    auto tempBytesCheck = sortByKeyTempStorage<KeyType, ValueType>(numElements);
    if (tempStorageBytes < tempBytesCheck) { throw std::runtime_error("temp storage too small\n"); };

    // Run sorting operation
    checkGpuErrors(cub::DeviceRadixSort::SortPairs(d_tempStorage, tempStorageBytes, d_keys, d_values, numElements));

    auto* curKeys = d_keys.Current();
    if (curKeys != first)
    {
        checkGpuErrors(cudaMemcpy(first, curKeys, numElements * sizeof(KeyType), cudaMemcpyDeviceToDevice));
    }

    auto* curValues = d_values.Current();
    if (curValues != values)
    {
        checkGpuErrors(cudaMemcpy(values, curValues, numElements * sizeof(ValueType), cudaMemcpyDeviceToDevice));
    }
}

#define SORT_BY_KEY_GPU_DB(KeyType, ValueType)                                                                         \
    template void sortByKeyGpu(KeyType*, KeyType*, ValueType*, KeyType*, ValueType*, void*, uint64_t);                 \
    template uint64_t sortByKeyTempStorage<KeyType, ValueType>(uint64_t)

SORT_BY_KEY_GPU_DB(unsigned, unsigned);
SORT_BY_KEY_GPU_DB(unsigned, int);
SORT_BY_KEY_GPU_DB(uint64_t, unsigned);
SORT_BY_KEY_GPU_DB(uint64_t, int);
SORT_BY_KEY_GPU_DB(uint64_t, uint64_t);
SORT_BY_KEY_GPU_DB(float, unsigned);

template<class KeyType, class ValueType>
void sortByKeyGpu(KeyType* first, KeyType* last, ValueType* values)
{
    thrust::sort_by_key(thrust::device, first, last, values);
}

template void sortByKeyGpu(unsigned*, unsigned*, unsigned*);
template void sortByKeyGpu(unsigned*, unsigned*, int*);
template void sortByKeyGpu(uint64_t*, uint64_t*, unsigned*);
template void sortByKeyGpu(uint64_t*, uint64_t*, int*);
template void sortByKeyGpu(uint64_t*, uint64_t*, uint64_t*);

template<class IndexType, class SumType>
void exclusiveScanGpu(const IndexType* first, const IndexType* last, SumType* output, SumType init)
{
    thrust::exclusive_scan(thrust::device, first, last, output, init);
}

template void exclusiveScanGpu(const int*, const int*, int*, int);
template void exclusiveScanGpu(const int*, const int*, unsigned*, unsigned);
template void exclusiveScanGpu(const int*, const int*, uint64_t*, uint64_t);
template void exclusiveScanGpu(const unsigned*, const unsigned*, unsigned*, unsigned);
template void exclusiveScanGpu(const unsigned*, const unsigned*, uint64_t*, uint64_t);

template<class ValueType>
size_t countGpu(const ValueType* first, const ValueType* last, ValueType v)
{
    return thrust::count(thrust::device, first, last, v);
}

template size_t countGpu(const int* first, const int* last, int v);
template size_t countGpu(const unsigned* first, const unsigned* last, unsigned v);
template size_t countGpu(const uint64_t* first, const uint64_t* last, uint64_t v);

} // namespace cstone
