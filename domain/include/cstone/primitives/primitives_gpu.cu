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

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>

#include "cstone/cuda/errorcheck.cuh"
#include "cstone/util/util.hpp"
#include "primitives_gpu.h"

namespace cstone
{

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

template void gatherGpu(const unsigned*, size_t, const int*, int*);
template void gatherGpu(const unsigned*, size_t, const util::array<float, 1>*, util::array<float, 1>*);
template void gatherGpu(const unsigned*, size_t, const util::array<float, 2>*, util::array<float, 2>*);
template void gatherGpu(const unsigned*, size_t, const util::array<float, 3>*, util::array<float, 3>*);
template void gatherGpu(const unsigned*, size_t, const util::array<float, 4>*, util::array<float, 4>*);

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

template<class T>
size_t lowerBoundGpu(const T* first, const T* last, T value)
{
    return thrust::lower_bound(thrust::device, first, last, value) - first;
}

template size_t lowerBoundGpu(const unsigned*, const unsigned*, unsigned);
template size_t lowerBoundGpu(const uint64_t*, const uint64_t*, uint64_t);
template size_t lowerBoundGpu(const int*, const int*, int);
template size_t lowerBoundGpu(const int64_t*, const int64_t*, int64_t);


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

template<class Tin, class Tout>
Tout reduceGpu(const Tin* input, size_t numElements, Tout init)
{
    return thrust::reduce(thrust::device, input, input + numElements, init);
}

template size_t reduceGpu(const unsigned*, size_t, size_t);

#include <thrust/iterator/zip_iterator.h>

template<class Tm, class Tv>
struct ekin
{
    HOST_DEVICE_FUN
    double operator()(const thrust::tuple<Tm, Tv, Tv, Tv>& vm)
    {
        cstone::Vec3<Tv> V{thrust::get<1>(vm), thrust::get<2>(vm), thrust::get<3>(vm)};
        return thrust::get<0>(vm) * norm2(V);
    }
};

template<class Tc, class Tv, class Tu, class Tm>
std::tuple<Tc, Tc, Vec3<Tc>, Vec3<Tc>> conservedQuantitiesGpu(const Tc* x,
                                                              const Tc* y,
                                                              const Tc* z,
                                                              const Tv* vx,
                                                              const Tv* vy,
                                                              const Tv* vz,
                                                              const Tu* u,
                                                              const Tm* m,
                                                              size_t first,
                                                              size_t last)
{
    auto it1 = thrust::make_zip_iterator(thrust::make_tuple(m + first, vx + first, vy + first, vz + first));
    auto it2 = thrust::make_zip_iterator(thrust::make_tuple(m + last, vx + last, vy + last, vz + last));

    double result = thrust::transform_reduce(thrust::device, it1, it2, ekin<Tm, Tv>{}, 0.0, thrust::plus<double>{});
    return {result, 0.0, Vec3<Tc>{0, 0, 0}, Vec3<Tc>{0, 0, 0}};
}

#define CONSERVED_Q_GPU(Tc, Tv, Tu, Tm)                                                                                \
    template std::tuple<Tc, Tc, Vec3<Tc>, Vec3<Tc>> conservedQuantitiesGpu(const Tc* x, const Tc* y, const Tc* z,      \
                                                                           const Tv* vx, const Tv* vy, const Tv* vz,   \
                                                                           const Tu* u, const Tm* m, size_t, size_t)

CONSERVED_Q_GPU(double, double, double, double);
CONSERVED_Q_GPU(float, float, float, float);

} // namespace cstone
