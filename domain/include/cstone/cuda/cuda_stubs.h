/*
 * MIT License
 *
 * Copyright (c) 2022 Politechnical University of Catalonia UPC
 *               2022 University of Basel
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
 * @brief  CUDA/Thrust stubs to provide declarations without definitions for use in non-CUDA builds
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <type_traits>
#include <vector>

template<class T, class Alloc>
T* rawPtr(std::vector<T, Alloc>& p)
{
    return p.data();
}

template<class T, class Alloc>
const T* rawPtr(const std::vector<T, Alloc>& p)
{
    return p.data();
}

template<class T>
void memcpyH2D(const T* src, size_t n, T* dest);

template<class T>
void memcpyD2H(const T* src, size_t n, T* dest);

template<class T>
void memcpyD2D(const T* src, size_t n, T* dest);

namespace thrust
{

template<class T>
class device_allocator;

template<class T, class Alloc>
class device_vector;

} // namespace thrust

/*! @brief detection trait to determine whether a template parameter is an instance of thrust::device_vector
 *
 * @tparam Vector the Vector type to check
 *
 * Add specializations for each type of vector that should be recognized as on device
 */
template<class Vector>
struct IsDeviceVector : public std::false_type
{
};

//! @brief detection of thrust device vectors
template<class T, class Alloc>
struct IsDeviceVector<thrust::device_vector<T, Alloc>> : public std::true_type
{
};
