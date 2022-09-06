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
 * @brief Thrust device vector instantiations
 *
 */

#include <thrust/device_vector.h>

//! @brief resizes a vector with a determined growth rate upon reallocation
template<class Vector>
void reallocateDevice(Vector& vector, size_t size, double growthRate)
{
    size_t current_capacity = vector.capacity();

    if (size > current_capacity)
    {
        size_t reserve_size = double(size) * growthRate;
        vector.reserve(reserve_size);
    }
    vector.resize(size);
}

template void reallocateDevice(thrust::device_vector<double>&, size_t, double);
template void reallocateDevice(thrust::device_vector<float>&, size_t, double);
template void reallocateDevice(thrust::device_vector<int>&, size_t, double);
template void reallocateDevice(thrust::device_vector<long>&, size_t, double);
template void reallocateDevice(thrust::device_vector<long long>&, size_t, double);
template void reallocateDevice(thrust::device_vector<unsigned>&, size_t, double);
template void reallocateDevice(thrust::device_vector<unsigned long>&, size_t, double);
template void reallocateDevice(thrust::device_vector<unsigned long long>&, size_t, double);
template void reallocateDevice(thrust::device_vector<char>&, size_t, double);

template<class Vector>
void reallocateDeviceShrink(Vector& vector, size_t size, double growthRate)
{
    size_t current_capacity = vector.capacity();

    if (size > current_capacity)
    {
        size_t reserve_size = double(size) * growthRate;
        vector.reserve(reserve_size);
    }
    vector.resize(size);
    if (double(vector.capacity()) / double(size) > growthRate * growthRate)
    {
        vector.shrink_to_fit();
    }
}

template void reallocateDeviceShrink(thrust::device_vector<double>&, size_t, double);
template void reallocateDeviceShrink(thrust::device_vector<float>&, size_t, double);
template void reallocateDeviceShrink(thrust::device_vector<int>&, size_t, double);
template void reallocateDeviceShrink(thrust::device_vector<long>&, size_t, double);
template void reallocateDeviceShrink(thrust::device_vector<long long>&, size_t, double);
template void reallocateDeviceShrink(thrust::device_vector<unsigned>&, size_t, double);
template void reallocateDeviceShrink(thrust::device_vector<unsigned long>&, size_t, double);
template void reallocateDeviceShrink(thrust::device_vector<unsigned long long>&, size_t, double);
template void reallocateDeviceShrink(thrust::device_vector<char>&, size_t, double);
