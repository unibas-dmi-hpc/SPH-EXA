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
 * @brief Thrust allocator adaptor to prevent value initialization
 *
 * Taken from: https://github.com/NVIDIA/thrust/blob/master/examples/uninitialized_vector.cu
 */

#pragma once

#include <thrust/device_allocator.h>
#include <thrust/device_vector.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <cassert>

namespace util
{
// uninitialized_allocator is an allocator which
// derives from device_allocator and which has a
// no-op construct member function
template<typename T>
struct uninitialized_allocator : thrust::device_allocator<T>
{
    // the default generated constructors and destructors are implicitly
    // marked __host__ __device__, but the current Thrust device_allocator
    // can only be constructed and destroyed on the host; therefore, we
    // define these as host only
    __host__ uninitialized_allocator() {}
    __host__ uninitialized_allocator(const uninitialized_allocator& other)
        : thrust::device_allocator<T>(other)
    {
    }
    __host__ ~uninitialized_allocator() {}

    uninitialized_allocator& operator=(const uninitialized_allocator&) = default;

    // for correctness, you should also redefine rebind when you inherit
    // from an allocator type; this way, if the allocator is rebound somewhere,
    // it's going to be rebound to the correct type - and not to its base
    // type for U
    template<typename U>
    struct rebind
    {
        typedef uninitialized_allocator<U> other;
    };

    // note that construct is annotated as
    // a __host__ __device__ function
    __host__ __device__ void construct(T*)
    {
        // no-op
    }
};

} // namespace util
