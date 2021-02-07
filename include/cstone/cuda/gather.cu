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

    DeviceMemory()
    {
        cudaStreamCreate(&streams_[0]);
        cudaStreamCreate(&streams_[1]);
    }

    ~DeviceMemory()
    {
        cudaStreamDestroy(streams_[0]);
        cudaStreamDestroy(streams_[1]);

        cudaFree(d_ordering_);

        cudaFreeHost(h_buffer_[0]);
        cudaFreeHost(h_buffer_[1]);

        cudaFree(d_buffer_[0]);
        cudaFree(d_buffer_[1]);
    }

    void reallocate(std::size_t newSize)
    {
        if (newSize > allocatedSize_)
        {
            // allocate 5% extra to avoid reallocation on small increase
            newSize = double(newSize) * 1.05;

            if (allocatedSize_ > 0)
            {
                cudaFree(d_ordering_);

                cudaFreeHost(h_buffer_[0]);
                cudaFreeHost(h_buffer_[1]);

                cudaFree(d_buffer_[0]);
                cudaFree(d_buffer_[1]);
            }

            cudaMalloc((void**)&d_ordering_,  newSize * sizeof(I));

            cudaHostAlloc((void**)&(h_buffer_[0]), newSize * sizeof(T), cudaHostAllocMapped);
            cudaHostAlloc((void**)&(h_buffer_[1]), newSize * sizeof(T), cudaHostAllocMapped);

            cudaMalloc((void**)&(d_buffer_[0]), newSize * sizeof(T));
            cudaMalloc((void**)&(d_buffer_[1]), newSize * sizeof(T));

            allocatedSize_ = newSize;
        }
    }

    I* ordering() { return d_ordering_; }

    T* stageBuffer(int i) { return h_buffer_[i]; }
    T* deviceBuffer(int i)      { return d_buffer_[i]; }

    cudaStream_t stream(int i) { return streams_[i]; }

private:
    std::size_t allocatedSize_{0} ;

    cudaStream_t streams_[2];

    //! \brief reorder map
    I* d_ordering_;
    //! \brief pinned host memory for zero-copy write-back
    T* h_buffer_[2];
    //! \brief device buffers
    T* d_buffer_[2];
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
DeviceGather<T, I>::~DeviceGather() = default;

template<class T, class I>
void DeviceGather<T, I>::operator()(T* values)
{
    // upload to device
    cudaMemcpy(deviceMemory_->deviceBuffer(0), values, mapSize_ * sizeof(T), cudaMemcpyHostToDevice);

    // reorder on device
    thrust::gather(thrust::device, deviceMemory_->ordering(), deviceMemory_->ordering() + mapSize_,
                   deviceMemory_->deviceBuffer(0), deviceMemory_->deviceBuffer(1));

    // download to host
    cudaMemcpy(values, deviceMemory_->deviceBuffer(1), mapSize_ * sizeof(T), cudaMemcpyDeviceToHost);
}

template<class T, class I>
void DeviceGather<T, I>::stage(const T* values)
{
    T* stageBuffer = deviceMemory_->stageBuffer(0);
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < mapSize_; ++i)
        stageBuffer[i] = values[i];
}

template<class T, class I>
void DeviceGather<T, I>::unstage(T* values)
{
    T* stageBuffer = deviceMemory_->stageBuffer(0);
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < mapSize_; ++i)
        values[i] = stageBuffer[i];
}

template<class T, class I>
void DeviceGather<T, I>::gatherStaged()
{
    T* d_stageBuffer;
    cudaHostGetDevicePointer((void**)&d_stageBuffer, deviceMemory_->stageBuffer(0), 0);
    // upload to device
    cudaMemcpy(deviceMemory_->deviceBuffer(0), deviceMemory_->stageBuffer(0),
               mapSize_ * sizeof(T), cudaMemcpyHostToDevice);

    // reorder on device and write back to host zero-copy memory
    thrust::gather(thrust::device, thrust::device_pointer_cast(deviceMemory_->ordering()),
                   thrust::device_pointer_cast(deviceMemory_->ordering() + mapSize_),
                   thrust::device_pointer_cast(deviceMemory_->deviceBuffer(0)),
                   thrust::device_pointer_cast(d_stageBuffer));
}

template<class T, class I>
void DeviceGather<T, I>::reorderArrays(T** arrays, int nArrays)
{
    // get the device pointer version for the pinned host stage buffers (zero-copy memory)
    T* stageBuffers[2];
    cudaHostGetDevicePointer((void**)&stageBuffers[0], deviceMemory_->stageBuffer(0), 0);
    cudaHostGetDevicePointer((void**)&stageBuffers[1], deviceMemory_->stageBuffer(1), 0);

    for (int i = 0; i < nArrays; ++i)
    {
        int streamID = i % 2;
        // upload array to device, this blocks on the host, since source not pinned
        // but allows operations in other streams on the GPU to run
        cudaMemcpyAsync(deviceMemory_->deviceBuffer(streamID), arrays[i],
                   mapSize_ * sizeof(T), cudaMemcpyHostToDevice,
                   deviceMemory_->stream(streamID));

        // reorder on device and write back to host zero-copy memory
        thrust::gather(thrust::cuda::par.on(deviceMemory_->stream(streamID)),
                       thrust::device_pointer_cast(deviceMemory_->ordering()),
                       thrust::device_pointer_cast(deviceMemory_->ordering() + mapSize_),
                       thrust::device_pointer_cast(deviceMemory_->deviceBuffer(streamID)),
                       thrust::device_pointer_cast(stageBuffers[streamID]));

        // move reordered data from host staging to original location
        cudaMemcpyAsync(arrays[i], deviceMemory_->stageBuffer(streamID), mapSize_ * sizeof(T),
                        cudaMemcpyHostToHost, deviceMemory_->stream(streamID));
    }
    cudaStreamSynchronize(deviceMemory_->stream(0));
    cudaStreamSynchronize(deviceMemory_->stream(1));
}

template class DeviceGather<float,  unsigned>;
template class DeviceGather<float,  uint64_t>;
template class DeviceGather<double, unsigned>;
template class DeviceGather<double, uint64_t>;
