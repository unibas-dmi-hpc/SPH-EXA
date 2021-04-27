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
 * @brief  General purpose utilities
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <utility>

#include "cstone/cuda/annotation.hpp"

/*! @brief A template to create structs as a type-safe version to using declarations
 *
 * Used in public API functions where a distinction between different
 * arguments of the same underlying type is desired. This provides a type-safe
 * version to using declarations. Instead of naming a type alias, the name
 * is used to define a struct that inherits from StrongType<T>, where T is
 * the underlying type.
 *
 * Due to the T() conversion and assignment from T,
 * an instance of StrongType<T> struct behaves essentially like an actual T, while construction
 * from T is disabled. This makes it impossible to pass a T as a function parameter
 * of type StrongType<T>.
 */
template<class T, class Phantom>
struct StrongType
{
    //! default ctor
    StrongType() : value_{} {}
    //! construction from the underlying type T, implicit conversions disabled
    explicit StrongType(T v) : value_(std::move(v)) {}

    //! assignment from T
    StrongType& operator=(T v)
    {
        value_ = std::move(v);
        return *this;
    }

    //! conversion to T
    operator T() const { return value_; }

    //! access the underlying value
    T value() const { return value_; }

private:
    T value_;
};

/*! @brief StrongType equality comparison
 *
 * Requires that both T and Phantom template parameters match.
 * For the case where a comparison between StrongTypes with matching T, but differing Phantom
 * parameters is desired, the underlying value attribute should be compared instead
 */
template<class T, class Phantom>
[[maybe_unused]] inline bool operator==(const StrongType<T, Phantom>& lhs, const StrongType<T, Phantom>& rhs)
{
    return lhs.value() == rhs.value();
}

//! comparison function <
template<class T, class Phantom>
inline bool operator<(const StrongType<T, Phantom>& lhs, const StrongType<T, Phantom>& rhs)
{
    return lhs.value() < rhs.value();
}

//! comparison function >
template<class T, class Phantom>
inline bool operator>(const StrongType<T, Phantom>& lhs, const StrongType<T, Phantom>& rhs)
{
    return lhs.value() > rhs.value();
}


//! @brief simple pair that's usable in both CPU and GPU code
template<class T>
class pair
{
public:
    pair() = default;

    CUDA_HOST_DEVICE_FUN
    pair(T first, T second) : data{first, second} {}

    CUDA_HOST_DEVICE_FUN       T& operator[](int i)       { return data[i]; }
    CUDA_HOST_DEVICE_FUN const T& operator[](int i) const { return data[i]; }

private:

    CUDA_HOST_DEVICE_FUN friend bool operator==(const pair& a, const pair& b)
    {
        return a.data[0] == b.data[0] && a.data[1] == b.data[1];
    }

    CUDA_HOST_DEVICE_FUN friend bool operator<(const pair& a, const pair& b)
    {
        bool c0 = a.data[0] < b.data[0];
        bool e0 = a.data[0] == b.data[0];
        bool c1 = a.data[1] < b.data[1];
        return c0 || (e0 && c1);
    }

    T data[2];
};


//! @brief ceil(divident/divisor) for integers
CUDA_HOST_DEVICE_FUN constexpr unsigned iceil(size_t dividend, unsigned divisor)
{
    return (dividend + divisor - 1) / divisor;
}