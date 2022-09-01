/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
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
 * @brief  Exposes gather functionality to reorder arrays by a map
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <vector>

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "errorcheck.cuh"
#include "gather.cuh"

namespace cstone
{

template<class LocalIndex>
class DeviceMemory
{
public:
    static constexpr size_t ElementSize = 8;

    DeviceMemory() = default;

    ~DeviceMemory()
    {
        if (allocatedSize_ > 0)
        {
            checkGpuErrors(cudaFree(d_ordering_));
        }
    }

    void reallocate(std::size_t newSize)
    {
        if (newSize > allocatedSize_)
        {
            // allocate 5% extra to avoid reallocation on small increase
            newSize = double(newSize) * 1.01;

            if (allocatedSize_ > 0)
            {
                checkGpuErrors(cudaFree(d_ordering_));
            }

            checkGpuErrors(cudaMalloc((void**)&d_ordering_, newSize * sizeof(LocalIndex)));

            allocatedSize_ = newSize;
        }
    }

    LocalIndex* ordering() { return d_ordering_; }

private:
    std::size_t allocatedSize_{0};

    //! @brief reorder map
    LocalIndex* d_ordering_;
};

template<class I>
__global__ void iotaKernel(I* buffer, size_t n, size_t offset)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) { buffer[tid] = offset + tid; }
}

template<class T, class I>
__global__ void reorder(I* map, const T* source, T* destination, size_t n)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) { destination[tid] = source[map[tid]]; }
}

template<class KeyType, class IndexType>
DeviceSfcSort<KeyType, IndexType>::DeviceSfcSort()
    : deviceMemory_(std::make_unique<DeviceMemory<IndexType>>())
{
}

template<class KeyType, class IndexType>
DeviceSfcSort<KeyType, IndexType>::~DeviceSfcSort() = default;

template<class KeyType, class IndexType>
const IndexType* DeviceSfcSort<KeyType, IndexType>::getReorderMap() const
{
    return deviceMemory_->ordering();
}

template<class KeyType, class IndexType>
void DeviceSfcSort<KeyType, IndexType>::setMapFromCodes(KeyType* firstKey, KeyType* lastKey)
{
    offset_     = 0;
    mapSize_    = lastKey - firstKey;
    numExtract_ = mapSize_;
    deviceMemory_->reallocate(mapSize_);

    unsigned numThreads = 256;
    unsigned numBlocks  = (mapSize_ + numThreads - 1) / numThreads;
    iotaKernel<<<numBlocks, numThreads>>>(deviceMemory_->ordering(), mapSize_, 0);
    checkGpuErrors(cudaGetLastError());

    // sort SFC keys on device, track new ordering on the device
    thrust::sort_by_key(thrust::device, firstKey, lastKey, deviceMemory_->ordering());
}

template<class KeyType, class IndexType>
template<class T>
void DeviceSfcSort<KeyType, IndexType>::operator()(const T* values,
                                                   T* destination,
                                                   IndexType offset,
                                                   IndexType numExtract) const
{
    static_assert(sizeof(T) <= DeviceMemory<IndexType>::ElementSize);

    unsigned numThreads = 256;
    unsigned numBlocks  = (numExtract + numThreads - 1) / numThreads;

    reorder<<<numBlocks, numThreads>>>(deviceMemory_->ordering() + offset, values, destination, numExtract);
    checkGpuErrors(cudaDeviceSynchronize());
}

template<class KeyType, class IndexType>
template<class T>
void DeviceSfcSort<KeyType, IndexType>::operator()(const T* values, T* destination) const
{
    this->operator()(values, destination, offset_, numExtract_);
}

template<class KeyType, class IndexType>
void DeviceSfcSort<KeyType, IndexType>::restrictRange(std::size_t offset, std::size_t numExtract)
{
    assert(offset + numExtract <= mapSize_);

    offset_     = offset;
    numExtract_ = numExtract;
}

template class DeviceSfcSort<unsigned, unsigned>;
template class DeviceSfcSort<uint64_t, unsigned>;

template void DeviceSfcSort<unsigned, unsigned>::operator()(const double*, double*) const;
template void DeviceSfcSort<unsigned, unsigned>::operator()(const float*, float*) const;

template void DeviceSfcSort<uint64_t, unsigned>::operator()(const double*, double*) const;
template void DeviceSfcSort<uint64_t, unsigned>::operator()(const float*, float*) const;

} // namespace cstone
