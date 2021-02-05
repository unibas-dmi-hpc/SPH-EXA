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

/*! \file
 * \brief  Exposes gather functionality to reorder arrays by a map
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <vector>

#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/sort.h>

#include "gather.cuh"

template<class T, class I>
class DeviceMemory
{
public:

    DeviceMemory() {}

    ~DeviceMemory()
    {
        cudaFreeHost(pinnedHostBuffer_);
        cudaFree(d_ordering_);
        cudaFree(d_source_);
        cudaFree(d_destination_);
    }

    void reallocate(std::size_t newSize)
    {
        if (newSize > allocatedSize_)
        {
            // allocate 5% extra to avoid reallocation on small increase
            newSize = double(newSize) * 1.05;

            if (allocatedSize_ > 0)
            {
                cudaFreeHost(pinnedHostBuffer_);
                cudaFree(d_ordering_);
                cudaFree(d_source_);
                cudaFree(d_destination_);
            }

            cudaHostAlloc((void**)&pinnedHostBuffer_, newSize * sizeof(T), cudaHostAllocMapped);
            cudaMalloc((void**)&d_ordering_,    newSize * sizeof(I));
            cudaMalloc((void**)&d_source_,      newSize * sizeof(T));
            cudaMalloc((void**)&d_destination_, newSize * sizeof(T));

            allocatedSize_ = newSize;
        }
    }

    T* stageBuffer() { return pinnedHostBuffer_; }

    I* ordering() { return d_ordering_; }

    T* source()      { return d_source_; }
    T* destination() { return d_destination_; }

private:
    std::size_t allocatedSize_{0} ;

    T* pinnedHostBuffer_;

    I* d_ordering_;
    T* d_source_;
    T* d_destination_;
};


template<class T, class I>
DeviceGather<T, I>::DeviceGather()
    : deviceMemory_(std::make_unique<DeviceMemory<T, I>>())
{}

template<class T, class I>
void DeviceGather<T, I>::setReorderMap(const I* map_first, const I* map_last)
{
    mapSize_      = map_last - map_first;
    deviceMemory_->reallocate(mapSize_);
    cudaMemcpy(deviceMemory_->ordering(), map_first, mapSize_ * sizeof(I), cudaMemcpyHostToDevice);
}

template<class T, class I>
DeviceGather<T, I>::~DeviceGather()
{
}

template<class T, class I>
void DeviceGather<T, I>::operator()(T* values)
{
    // upload to device
    cudaMemcpy(deviceMemory_->source(), values, mapSize_ * sizeof(T), cudaMemcpyHostToDevice);

    // reorder on device
    thrust::gather(thrust::device, deviceMemory_->ordering(), deviceMemory_->ordering() + mapSize_,
                   deviceMemory_->source(), deviceMemory_->destination());

    // download to host
    cudaMemcpy(values, deviceMemory_->destination(), mapSize_ * sizeof(T), cudaMemcpyDeviceToHost);
}

template<class T, class I>
void DeviceGather<T, I>::stage(const T* values)
{
    T* stageBuffer = deviceMemory_->stageBuffer();
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < mapSize_; ++i)
        stageBuffer[i] = values[i];
}

template<class T, class I>
void DeviceGather<T, I>::unstage(T* values)
{
    T* stageBuffer = deviceMemory_->stageBuffer();
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < mapSize_; ++i)
        values[i] = stageBuffer[i];
}

template<class T, class I>
void DeviceGather<T, I>::gatherStaged()
{
    T* d_stageBuffer;
    cudaHostGetDevicePointer((void**)&d_stageBuffer, deviceMemory_->stageBuffer(), 0);
    // upload to device
    cudaMemcpy(deviceMemory_->source(), deviceMemory_->stageBuffer(), mapSize_ * sizeof(T), cudaMemcpyHostToDevice);

    // reorder on device and write back to host zero-copy memory
    thrust::gather(thrust::device, thrust::device_pointer_cast(deviceMemory_->ordering()),
                   thrust::device_pointer_cast(deviceMemory_->ordering() + mapSize_),
                   thrust::device_pointer_cast(deviceMemory_->source()), thrust::device_pointer_cast(d_stageBuffer));
}

template class DeviceGather<float,  unsigned>;
template class DeviceGather<float,  uint64_t>;
template class DeviceGather<double, unsigned>;
template class DeviceGather<double, uint64_t>;
