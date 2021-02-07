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
#include <thrust/sort.h>

#include "gather.cuh"

template<class T, class I>
class DeviceMemory
{
public:

    DeviceMemory() = default;

    ~DeviceMemory()
    {
        if (allocatedSize_ > 0)
        {
            cudaFree(d_ordering_);

            cudaFree(d_buffer_[0]);
            cudaFree(d_buffer_[1]);
        }
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

                cudaFree(d_buffer_[0]);
                cudaFree(d_buffer_[1]);
            }

            cudaMalloc((void**)&d_ordering_,  newSize * sizeof(I));

            cudaMalloc((void**)&(d_buffer_[0]), newSize * sizeof(T));
            cudaMalloc((void**)&(d_buffer_[1]), newSize * sizeof(T));

            allocatedSize_ = newSize;
        }
    }

    I* ordering() { return d_ordering_; }

    T* deviceBuffer(int i)      { return d_buffer_[i]; }

private:
    std::size_t allocatedSize_{0} ;

    //! \brief reorder map
    I* d_ordering_;
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
__global__ void reorder(I* map, T* source, T* destination, size_t n)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid > n) return;

    destination[tid] = source[map[tid]];
}

template<class T, class I>
void DeviceGather<T, I>::operator()(T* values)
{
    constexpr int nThreads = 256;
    int nBlocks = (mapSize_ + nThreads - 1) / nThreads;

    // upload to device
    cudaMemcpy(deviceMemory_->deviceBuffer(0), values, mapSize_ * sizeof(T), cudaMemcpyHostToDevice);

    // reorder on device
    reorder<<<nBlocks, nThreads>>>(deviceMemory_->ordering(),
                                   deviceMemory_->deviceBuffer(0),
                                   deviceMemory_->deviceBuffer(1),
                                   mapSize_);

    // download to host
    cudaMemcpy(values, deviceMemory_->deviceBuffer(1), mapSize_ * sizeof(T), cudaMemcpyDeviceToHost);
}

template class DeviceGather<float,  unsigned>;
template class DeviceGather<float,  uint64_t>;
template class DeviceGather<double, unsigned>;
template class DeviceGather<double, uint64_t>;
