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
 * @brief Reallocation of thrust device vectors in a separate compilation unit for use from .cpp code
 */

#pragma once

template<class Vector>
extern void reallocateDevice(Vector&, size_t, double);

/*! @brief resize a device vector to given number of bytes if current size is smaller
 *
 * @param[inout] vec       a device vector like thrust::device_vector
 * @param[in]    numBytes  minimum buffer size in bytes of @a vec
 * @return                 number of elements (vec.size(), not bytes) of supplied argument vector
 */
template<class Vector>
size_t reallocateDeviceBytes(Vector& vec, size_t numBytes)
{
    constexpr size_t elementSize = sizeof(typename Vector::value_type);
    size_t originalSize          = vec.size();

    size_t currentSizeBytes = originalSize * elementSize;
    if (currentSizeBytes < numBytes) { reallocateDevice(vec, (numBytes + elementSize - 1) / elementSize, 1.01); }

    return originalSize;
}

//! @brief resizes a vector with a determined growth rate upon reallocation
template<class Vector>
void reallocate(Vector& vector, size_t size, double growthRate)
{
    size_t current_capacity = vector.capacity();

    if (size > current_capacity)
    {
        size_t reserve_size = double(size) * growthRate;
        vector.reserve(reserve_size);
    }
    vector.resize(size);
}

template<class... Arrays>
void reallocate(std::size_t size, Arrays&... arrays)
{
    [[maybe_unused]] std::initializer_list<int> list{(reallocate(arrays, size, 1.01), 0)...};
}
