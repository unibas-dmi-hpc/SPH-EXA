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

#include "cstone/cuda/cuda_stubs.h"
#include "cstone/util/tuple_util.hpp"

template<class Vector>
extern void reallocateDevice(Vector&, size_t, double);

template<class Vector>
extern void reallocateDeviceShrink(Vector&, size_t, double, double);

//! @brief resizes a vector with a determined growth rate upon reallocation
template<class Vector>
void reallocateGeneric(Vector& vector, size_t size, double growthRate)
{
    size_t current_capacity = vector.capacity();

    if (size > current_capacity)
    {
        size_t reserve_size = double(size) * growthRate;
        vector.reserve(reserve_size);
    }
    vector.resize(size);
}

template<class Vector, std::enable_if_t<IsDeviceVector<Vector>{}, int> = 0>
void reallocate(Vector& vector, size_t size, double growthRate)
{
    reallocateDevice(vector, size, growthRate);
}

template<class Vector, std::enable_if_t<!IsDeviceVector<Vector>{}, int> = 0>
void reallocate(Vector& vector, size_t size, double growthRate)
{
    reallocateGeneric(vector, size, growthRate);
}

//! @brief if reallocation of the underlying buffer is necessary, first deallocate it
template<class Vector>
void reallocateDestructive(Vector& vector, size_t size, double growthRate)
{
    if (size > vector.capacity())
    {
        // swap with an empty temporary to force deallocation
        Vector().swap(vector);
    }
    reallocate(vector, size, growthRate);
}

template<class... Arrays>
void reallocate(std::size_t size, Arrays&... arrays)
{
    [[maybe_unused]] std::initializer_list<int> list{(reallocate(arrays, size, 1.01), 0)...};
}

/*! @brief resize a vector to given number of bytes if current size is smaller
 *
 * @param[inout] vec       an STL or thrust-like vector
 * @param[in]    numBytes  minimum buffer size in bytes of @a vec
 * @return                 number of elements (vec.size(), not bytes) of supplied argument vector
 *
 * Note: previous content is destroyed
 */
template<class Vector>
size_t reallocateBytes(Vector& vec, size_t numBytes)
{
    constexpr size_t elementSize = sizeof(typename Vector::value_type);
    size_t originalSize          = vec.size();

    size_t currentSizeBytes = originalSize * elementSize;
    if (currentSizeBytes < numBytes) { reallocateDestructive(vec, (numBytes + elementSize - 1) / elementSize, 1.01); }

    return originalSize;
}

//! @brief reallocate memory by first deallocating all scratch to reduce fragmentation and decrease temp mem footprint
template<class... Vectors1, class... Vectors2>
void lowMemReallocate(size_t size,
                      double growthFactor,
                      std::tuple<Vectors1&...> conserved,
                      std::tuple<Vectors2&...> scratch)
{
    // if the new size exceeds capacity, we first deallocate all scratch buffers to make space for the reallocations
    util::for_each_tuple(
        [size](auto& v)
        {
            if (size > v.capacity()) { std::decay_t<decltype(v)>{}.swap(v); }
        },
        scratch);
    util::for_each_tuple([size, growthFactor](auto& v) { reallocate(v, size, growthFactor); }, conserved);
    util::for_each_tuple([size, growthFactor](auto& v) { reallocate(v, size, growthFactor); }, scratch);
}
