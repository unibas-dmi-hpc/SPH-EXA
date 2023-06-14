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
 * @brief  Functionality that exist in std::, but cannot be used in device code
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <type_traits>
#include "cstone/cuda/annotation.hpp"

namespace stl
{

template<typename T, T v>
struct integral_constant
{
    static constexpr T value = v;
    typedef T value_type;
    typedef integral_constant<T, v> type;

    HOST_DEVICE_FUN
    constexpr operator value_type() const noexcept { return value; } // NOLINT
};

//! @brief This does what you think it does
template<class T>
HOST_DEVICE_FUN constexpr const T& min(const T& a, const T& b)
{
    if (b < a) return b;
    return a;
}

//! @brief This does what you think it does
template<class T>
HOST_DEVICE_FUN constexpr const T& max(const T& a, const T& b)
{
    if (a < b) return b;
    return a;
}

//! @brief the std version is not constexpr, this here requires two's complement
template<class T>
HOST_DEVICE_FUN constexpr std::enable_if_t<std::is_signed_v<T>, T> abs(T a)
{
#ifdef __CUDA_ARCH__
    if constexpr (std::is_same_v<T, int>) { return ::abs(a); }
    else { return ::labs(a); }
#else
    T mask = a >> (sizeof(T) * 8 - 1);
    return (a ^ mask) - mask;
#endif
}

//! @brief a simplified version of std::lower_bound that can be compiled as device code
template<class ForwardIt, class T>
HOST_DEVICE_FUN ForwardIt lower_bound(ForwardIt first, ForwardIt last, const T& value)
{
    ForwardIt it;
    long long int step;
    long long int count = last - first;

    while (count > 0)
    {
        it   = first;
        step = count / 2;
        it += step;
        if (*it < value)
        {
            first = ++it;
            count -= step + 1;
        }
        else
            count = step;
    }
    return first;
}

//! @brief a simplified version of std::upper_bound that can be compiled as device code
template<class ForwardIt, class T>
HOST_DEVICE_FUN ForwardIt upper_bound(ForwardIt first, ForwardIt last, const T& value)
{
    ForwardIt it;
    long long int step;
    long long int count = last - first;

    while (count > 0)
    {
        it   = first;
        step = count / 2;
        it += step;
        if (!(value < *it)) // NOLINT
        {
            first = ++it;
            count -= step + 1;
        }
        else
            count = step;
    }
    return first;
}

} // namespace stl
